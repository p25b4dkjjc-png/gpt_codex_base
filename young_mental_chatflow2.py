from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import httpx
import json
import uvicorn
from datetime import datetime
import logging

# 创建 FastAPI 应用
app = FastAPI(title="AI Mental ChatFlow", version="1.0.0")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
API_BASE_URL = "http://192.168.0.2:8080/v1"
API_KEY = "app-H9TluHM63V7CnI6beuO6ALdv"
USER_ID = "abc-123"

# 请求头
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream"
}


# 错误类型定义
class ErrorType:
    NETWORK_ERROR = "NETWORK_ERROR"  # 网络异常
    SERVICE_ERROR = "SERVICE_ERROR"  # 服务异常
    DATA_ERROR = "DATA_ERROR"  # 数据异常
    BUSINESS_ERROR = "BUSINESS_ERROR"  # 业务异常
    UNKNOWN_ERROR = "UNKNOWN_ERROR"  # 未知异常


# 错误码定义
class ErrorCode:
    # 网络相关错误 1xxx
    NETWORK_TIMEOUT = 1001
    NETWORK_CONNECTION_FAILED = 1002

    # 服务相关错误 2xxx
    DIFY_SERVICE_UNAVAILABLE = 2001
    DATABASE_UNAVAILABLE = 2003

    # 数据相关错误 3xxx
    INVALID_REQUEST_DATA = 3001
    CONVERSATION_NOT_FOUND = 3002

    # 未知错误 9xxx
    UNKNOWN_SYSTEM_ERROR = 9001


# 统一错误响应模型
class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    error_code: int
    error_message: str
    detail: Optional[str] = None
    retry_count: Optional[int] = None
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 错误处理工具函数
def create_error_response(error_type: str, error_code: int, error_message: str,
                          detail: Optional[str] = None, retry_count: Optional[int] = None) -> ErrorResponse:
    """创建统一的错误响应"""
    return ErrorResponse(
        error_type=error_type,
        error_code=error_code,
        error_message=error_message,
        detail=detail,
        retry_count=retry_count
    )


def create_streaming_error(error_type: str, error_code: int, error_message: str,
                           detail: Optional[str] = None, retry_count: Optional[int] = None) -> str:
    """创建流式接口的错误响应"""
    error_response = create_error_response(error_type, error_code, error_message, detail, retry_count)
    return f"data: {json.dumps(error_response.model_dump(), ensure_ascii=False)}\n"


def classify_exception(e: Exception) -> tuple[str, int, str]:
    """异常分类，返回错误类型、错误码、错误信息"""
    error_str = str(e)

    # 网络相关异常
    if any(keyword in error_str.lower() for keyword in ['timeout', 'connection', 'network']):
        if 'timeout' in error_str.lower():
            return ErrorType.NETWORK_ERROR, ErrorCode.NETWORK_TIMEOUT, "网络请求超时"
        else:
            return ErrorType.NETWORK_ERROR, ErrorCode.NETWORK_CONNECTION_FAILED, "网络连接失败"

    # HTTP相关异常
    elif any(keyword in error_str.lower() for keyword in ['404', 'not found']):
        return ErrorType.DATA_ERROR, ErrorCode.CONVERSATION_NOT_FOUND, "对话数据不存在"
    elif any(keyword in error_str.lower() for keyword in ['500', 'internal server error']):
        return ErrorType.SERVICE_ERROR, ErrorCode.DIFY_SERVICE_UNAVAILABLE, "Dify服务暂时不可用"

    # 数据解析异常
    elif any(keyword in error_str.lower() for keyword in ['json', 'parse', 'decode']):
        return ErrorType.DATA_ERROR, ErrorCode.INVALID_REQUEST_DATA, "数据解析失败"

    # 默认未知异常
    else:
        return ErrorType.UNKNOWN_ERROR, ErrorCode.UNKNOWN_SYSTEM_ERROR, "系统异常，请稍后重试"


class ProcessRequest(BaseModel):
    role: Optional[int] = 0
    query: str
    conversation_id: str  # 会话ID，为空时创建新会话


async def process_streaming_response(response, query: str = ""):
    """处理 Chatflow 流式响应"""
    full_answer = ""
    conversation_id = None
    created_at = None
    current_branch = None
    current_intent = None
    start_event_sent = False

    # ✅ 新增：标记第一个消息和缓存最后一个消息
    first_message = True
    pending_message = None  # 用于缓存上一个 message 事件
    text_buffer = ""  # 文本聚合缓冲，减少过碎切片导致的 TTS 卡顿

    # TTS 友好的分段参数（只改变分段策略，不改变输出结构）
    TTS_MIN_CHARS = 18
    TTS_MAX_CHARS = 70
    TTS_SOFT_PUNCT = "，。！？；：,.!?;:"

    # 测试模式标记
    test_trigger = query == "1qaz@WSX#EDC"
    is_test_processed = False

    BRANCH_MAP = {
        '1774422149518': {'type': 'risk_or_manual', 'interviewStatus': True, 'dialogRound': 1, 'desc': '风险/人工服务'},
        '1774422187822': {'type': 'intro', 'interviewStatus': False, 'dialogRound': 0, 'desc': '12355介绍'},
        '1774489056273': {'type': 'greeting', 'interviewStatus': False, 'dialogRound': 0, 'desc': '寒暄/结束语'},
        'answer': {'type': 'fallback', 'interviewStatus': False, 'dialogRound': 0, 'desc': '兜底回复'},
        '1774489837177': {'type': 'consultation', 'interviewStatus': True, 'dialogRound': 1, 'desc': '问题咨询'}
    }

    def emit_text_message(text: str):
        """构造统一 message 事件（保持原有数据结构）"""
        if not text:
            return None
        simplified = {
            "event": "message",
            "conversation_id": conversation_id,
            "created_at": created_at,
            "intent": current_intent,
            "answer": text
        }
        return f"data: {json.dumps(simplified, ensure_ascii=False)}\n\n"

    def split_tts_friendly_segments(text: str):
        """按语义标点优先切片，避免极短碎片造成语音卡顿"""
        segments = []
        while len(text) >= TTS_MIN_CHARS:
            search_end = min(len(text), TTS_MAX_CHARS)
            split_index = -1

            for idx in range(search_end - 1, -1, -1):
                if text[idx] in TTS_SOFT_PUNCT and idx + 1 >= TTS_MIN_CHARS:
                    split_index = idx + 1
                    break

            if split_index == -1:
                if len(text) >= TTS_MAX_CHARS:
                    split_index = TTS_MAX_CHARS
                else:
                    break

            segment = text[:split_index]
            segments.append(segment)
            text = text[split_index:]

        return segments, text

    async for line in response.aiter_lines():
        if not line or line.startswith(":"):
            continue

        if line.startswith("data:"):
            try:
                json_str = line[5:].strip()
                if not json_str:
                    continue

                json_data = json.loads(json_str)
                event_type = json_data.get('event')

                if json_data.get('conversation_id'):
                    conversation_id = json_data['conversation_id']
                if json_data.get('created_at'):
                    created_at = json_data['created_at']

                if event_type == 'message':
                    answer = json_data.get('answer', '')

                    # ✅ 关键修改：处理第一个消息的开头换行符
                    if answer and first_message:
                        answer = answer.lstrip('\n')
                        first_message = False

                    # ✅ 关键修改：如果有缓存的上一个消息，先发送它（此时确认它不是最后一个）
                    if pending_message is not None:
                        yield pending_message
                        pending_message = None

                    # 测试模式逻辑保持不变
                    if test_trigger and answer and not is_test_processed:
                        is_test_processed = True
                        full_text = answer
                        chunk_size = max(1, len(full_text) // 5)

                        for i in range(4):
                            start = i * chunk_size
                            end = start + chunk_size if i < 3 else len(full_text) - chunk_size
                            chunk_text = full_text[start:end]

                            # ✅ 测试模式也处理第一个和最后一个 chunk 的换行符
                            if i == 0:
                                chunk_text = chunk_text.lstrip('\n')
                            if i == 3:  # 最后一个 chunk（第4个，索引3）
                                chunk_text = chunk_text.rstrip('\n')

                            if chunk_text:
                                simplified = {
                                    "event": "message",
                                    "conversation_id": conversation_id,
                                    "created_at": created_at,
                                    "intent": current_intent,
                                    "answer": chunk_text
                                }
                                yield f"data: {json.dumps(simplified, ensure_ascii=False)}\n\n"

                        error_data = {
                            "event": "Error",
                            "conversation_id": conversation_id or "",
                            "created_at": created_at,
                            "intent": 0,
                            "answer": "模拟测试异常：第5轮流式输出中断"
                        }
                        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                        return

                    # 正常模式
                    if answer:
                        full_answer += answer

                        # 首个消息块到达时立即发送 node_started（如未发送）
                        if not start_event_sent and conversation_id:
                            start_event = {
                                "event": "node_started",
                                "conversation_id": conversation_id,
                                "created_at": created_at,
                                "intent": current_intent
                            }
                            yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
                            start_event_sent = True

                        text_buffer += answer
                        ready_segments, text_buffer = split_tts_friendly_segments(text_buffer)

                        for segment in ready_segments:
                            # 有新片段可发时，先把上一个已确认非末尾片段发出
                            if pending_message is not None:
                                yield pending_message
                            # 缓存当前消息（此时不确定它是否是最后一个）
                            pending_message = emit_text_message(segment)

                elif event_type == 'node_finished':
                    data = json_data.get('data') or {}
                    node_type = data.get('node_type')
                    node_id = data.get('node_id')
                    outputs = data.get('outputs') or {}

                    if node_id == '1774421871735' and node_type == 'llm':
                        intent_text = outputs.get('text', '').strip()
                        if intent_text in ['1', '2', '3', '4']:
                            current_intent = int(intent_text)

                    elif node_type == 'answer' and node_id in BRANCH_MAP:
                        if not test_trigger:
                            current_branch = BRANCH_MAP[node_id]

                            # node_finished 前先把残余文本作为最后片段输出
                            if text_buffer:
                                pending_message = emit_text_message(text_buffer)
                                text_buffer = ""

                            # ✅ 关键修改：在发送 node_finished 之前，处理并发送最后一个 message
                            if pending_message is not None:
                                try:
                                    msg_json_str = pending_message[5:].strip()
                                    msg_data = json.loads(msg_json_str)
                                    if msg_data.get('answer', '').endswith('\n'):
                                        msg_data['answer'] = msg_data['answer'].rstrip('\n')

                                    # 修复：去掉换行符后如果为空，就不输出
                                    if msg_data.get('answer'):
                                        pending_message = f"data: {json.dumps(msg_data, ensure_ascii=False)}\n\n"
                                        yield pending_message
                                except:
                                    yield pending_message
                                pending_message = None

                            final_event = {
                                "event": "node_finished",
                                "conversation_id": conversation_id,
                                "created_at": created_at,
                                "intent": current_intent,
                                "answer": ""
                            }
                            yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"

                elif event_type == 'error':
                    # error 前先把残余文本作为最后片段输出
                    if text_buffer:
                        pending_message = emit_text_message(text_buffer)
                        text_buffer = ""

                    # ✅ 关键修改：在发送 error 事件之前，先处理并发送最后一个 message（如果存在）
                    if pending_message is not None:
                        try:
                            msg_json_str = pending_message[5:].strip()
                            msg_data = json.loads(msg_json_str)
                            if msg_data.get('answer', '').endswith('\n'):
                                msg_data['answer'] = msg_data['answer'].rstrip('\n')
                            pending_message = f"data: {json.dumps(msg_data, ensure_ascii=False)}\n\n"
                        except:
                            pass
                        yield pending_message
                        pending_message = None

                    error_event = {
                        "event": "error",
                        "conversation_id": conversation_id,
                        "code": json_data.get('code'),
                        "message": json_data.get('message')
                    }
                    yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

            except json.JSONDecodeError:
                continue
            except Exception as e:
                yield f"data: {json.dumps({'event': 'parse_error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    # ✅ 关键修改：流结束后的清理 - 处理最后一个 message（如果流意外结束没有 node_finished）
    if text_buffer:
        pending_message = emit_text_message(text_buffer)
        text_buffer = ""

    if pending_message is not None:
        try:
            msg_json_str = pending_message[5:].strip()
            msg_data = json.loads(msg_json_str)
            if msg_data.get('answer', '').endswith('\n'):
                msg_data['answer'] = msg_data['answer'].rstrip('\n')
            pending_message = f"data: {json.dumps(msg_data, ensure_ascii=False)}\n\n"
        except:
            pass
        yield pending_message


def classify_exception_fun(e: Exception) -> str:
    """异常分类，返回错误类型、错误码、错误信息"""
    error_str = str(e)

    # 网络相关异常
    if any(keyword in error_str.lower() for keyword in ['timeout', 'connection', 'network']):
        if 'timeout' in error_str.lower():
            return "网络请求超时"
        else:
            return "网络连接失败"

    # 数据库相关异常
    elif any(keyword in error_str.lower() for keyword in ['mysql', 'database', 'connection refused']):
        return "数据库服务不可用"

    # HTTP相关异常
    elif any(keyword in error_str.lower() for keyword in ['404', 'not found']):
        return "对话数据不存在"
    elif any(keyword in error_str.lower() for keyword in ['500', 'internal server error']):
        return "Dify服务暂时不可用"

    # 数据解析异常
    elif any(keyword in error_str.lower() for keyword in ['json', 'parse', 'decode']):
        return "数据解析失败"
    elif any(keyword in error_str.lower() for keyword in ['400', 'invalid_param']):
        return "传入参数异常"
    elif any(keyword in error_str.lower() for keyword in ['400', 'app_unavailable']):
        return "App 配置不可用"
    elif any(keyword in error_str.lower() for keyword in ['400', 'provider_not_initialize']):
        return "无可用模型凭据配置"
    elif any(keyword in error_str.lower() for keyword in ['400', 'provider_quota_exceeded']):
        return "模型调用额度不足"
    elif any(keyword in error_str.lower() for keyword in ['400', 'model_currently_not_support']):
        return "当前模型不可用"
    elif any(keyword in error_str.lower() for keyword in ['400', 'workflow_request_error']):
        return "workflow 执行失败"
    elif any(keyword in error_str.lower() for keyword in ['400', 'completion_request_error']):
        return "文本生成失败"
    # 默认未知异常
    else:
        return "系统异常，请稍后重试"


async def create_interview(query):
    """创建会话"""
    try:
        # 构建请求载荷 - 改为 blocking 模式
        payload = {
            "inputs": {},
            "query": query,
            "response_mode": "blocking",  # 改为 blocking 模式
            "conversation_id": "",
            "user": USER_ID,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/chat-messages",
                headers=HEADERS,
                json=payload
            )

            if response.status_code != 200:
                error_message = classify_exception_fun(
                    Exception(f"HTTP状态码: {response.status_code}")
                )
                logging.error(f"request failed:    {response.status_code}   {error_message} ")
                return {
                    "code": response.status_code,
                    "conversation_id": '',
                    "message": error_message
                }

            # 解析完整的响应数据
            response_data = response.json()
            data = {}
            data['conversation_id'] = response_data.get("conversation_id", "")
            # 提取关键信息
            result = {
                "code": 200,
                "data": data,
                "message": ''
            }

            return result

    except Exception as e:
        error_message = classify_exception_fun(e)
        logging.error(f"request failed:    {500}   {error_message} {e}")
        return {
            "code": 500,
            "conversation_id": '',
            "message": error_message
        }


@app.get("/api/young/create_conversation")
async def create_intent_conversation():
    """创建会话"""
    logging.info("----------------创建会话----------------")
    result = await create_interview("创建会话")
    logging.info(f"请求结束获取返回结果--->{result}")
    return result


@app.post("/api/mental/process")
async def process_mental_chat(request: ProcessRequest):
    """AI 心理咨询对话接口 - 支持工作流模式流式输出

    Args:
        request: 包含 role、query 和 conversation_id 的请求体
                conversation_id 为空时返回流式错误
    """
    try:
        # 处理用户输入
        query = request.query
        role = request.role
        conv_id = request.conversation_id if request.conversation_id else ""

        # ✅ 关键修改：在建立 Dify 连接前判断会话ID是否为空
        if not conv_id:
            error_message = "会话ID不能为空，请先调用 /api/mental/create_conversation 创建会话"
            logging.error(f"request failed: 400 {error_message}")

            # 返回流式格式的错误（保持接口一致性）
            async def error_stream():
                error_data = {
                    "event": "Error",
                    "conversation_id": conv_id or "",
                    "created_at": 0,
                    "intent": 0,
                    "answer": error_message
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        # 构建请求载荷 - 工作流模式
        payload = {
            "inputs": {
                "role": role
            },
            "query": query,
            "response_mode": "streaming",
            "conversation_id": conv_id,
            "user": USER_ID,
        }
        logger.info(f"Processing chat request - conversation_id: {conv_id}, query: {query[:20]}...")

        # 创建生成器函数来处理流式响应
        async def generate_response():
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream(
                            "POST",
                            f"{API_BASE_URL}/chat-messages",
                            headers=HEADERS,
                            json=payload
                    ) as response:

                        if response.status_code != 200:
                            error_detail = ""
                            try:
                                error_body = await response.aread()
                                error_detail = error_body.decode('utf-8', errors='ignore')
                            except:
                                pass

                            # ✅ 使用 classify_exception_fun 保持与 create_interview 一致
                            error_message = classify_exception_fun(
                                Exception(f"HTTP状态码: {response.status_code}")
                            )
                            logging.error(f"request failed: {response.status_code} {error_message} {error_detail}")

                            # 统一错误格式
                            error_data = {
                                "event": "Error",
                                "conversation_id": conv_id or "",
                                "created_at": 0,
                                "intent": 0,
                                "answer": error_message
                            }
                            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                            return

                        async for chunk in process_streaming_response(response, query):
                            yield chunk

            except httpx.TimeoutException as timeout_e:
                # ✅ 统一使用 classify_exception_fun
                error_message = classify_exception_fun(timeout_e)
                logging.error(f"request failed: 500 {error_message} {timeout_e}")
                error_data = {
                    "event": "Error",
                    "conversation_id": conv_id or "",
                    "created_at": 0,
                    "intent": 0,
                    "answer": error_message
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

            except Exception as stream_e:
                error_message = classify_exception_fun(stream_e)
                logging.error(f"request failed: 500 {error_message} {stream_e}")
                error_data = {
                    "event": "Error",
                    "conversation_id": conv_id or "",
                    "created_at": 0,
                    "intent": 0,
                    "answer": error_message
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        # 顶层异常也保持格式一致
        error_message = classify_exception_fun(e)
        logging.error(f"request failed: 500 {error_message} {e}")

        # 返回流式格式的错误
        async def error_stream():
            error_data = error_data = {
                "event": "Error",
                "conversation_id": conv_id or "",
                "created_at": 0,
                "intent": 0,
                "answer": error_message
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "message": "AI Mental ChatFlow API is running"}


# 主程序入口
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8213,
        log_level="info"
    )
