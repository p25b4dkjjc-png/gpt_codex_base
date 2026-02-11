import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Tuple, Optional, Union, List
import tempfile
import shutil
import cv2
import dashscope
import numpy as np
import pdfplumber
import requests
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, Function, utility  # type: ignore
from rapidocr_onnxruntime import RapidOCR
import logging
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import platform

MILVUS_HOST = "192.168.0.2"
MILVUS_PORT = "19530"
COLLECTION_NAME = "job_recommendations"

DASHSCOPE_API_KEY = "sk-b8855f2599ea4111a20e3b621713e97b"
QWEN_API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("<"):
    raise RuntimeError("未提供 DashScope API Key。请在代码中配置 DASHSCOPE_API_KEY。")
dashscope.api_key = DASHSCOPE_API_KEY

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ Windows优化: 使用ProactorEventLoop（Python 3.8+默认） ============
if platform.system() == 'Windows':
    # Windows下确保使用ProactorEventLoop（高性能IOCP实现）
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    logger.info("Windows系统检测到，使用ProactorEventLoop（IOCP）")

# ============ 优化1: 预初始化OCR模型 + 线程池 ============
_ocr_instance: Optional[RapidOCR] = None
_thread_pool: Optional[ThreadPoolExecutor] = None
_semaphore: Optional[asyncio.Semaphore] = None  # 限制并发数

# 连接池复用
_aiohttp_session: Optional[aiohttp.ClientSession] = None


def get_ocr_instance() -> RapidOCR:
    """ocr模型初始化（启动时预加载）"""
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("正在初始化OCR模型...")
        start = time.time()
        _ocr_instance = RapidOCR()
        logger.info(f"OCR模型初始化完成，耗时: {time.time() - start:.2f}s")
    return _ocr_instance


def get_thread_pool() -> ThreadPoolExecutor:
    """获取线程池"""
    global _thread_pool
    if _thread_pool is None:
        # Windows下建议workers不要设置太大，避免GIL竞争
        max_workers = min(16, (os.cpu_count() or 1) * 2)
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ocr_worker")
        logger.info(f"线程池初始化完成， workers: {max_workers}")
    return _thread_pool


def get_semaphore() -> asyncio.Semaphore:
    """获取并发控制信号量"""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(3)  # Windows下保守设置，避免内存压力
    return _semaphore


async def get_session() -> aiohttp.ClientSession:
    """获取复用的aiohttp session"""
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        # Windows下TCP连接优化
        connector = aiohttp.TCPConnector(
            limit=10,  # Windows下适当降低连接数
            limit_per_host=5,  # 单域名连接限制
            ttl_dns_cache=300,  # DNS缓存5分钟
            use_dns_cache=True,
            enable_cleanup_closed=True,  # 清理关闭的连接
            force_close=False,  # 允许连接复用
        )
        _aiohttp_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json"
            }
        )
    return _aiohttp_session


def new_milvus_collection() -> Collection:
    """Connect to Milvus and return the job collection"""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)
    try:
        col.load()
    except Exception:
        pass
    return col


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - 预初始化所有重资源
    global _milvus_collection

    logger.info("=== 服务启动中，预初始化资源 ===")

    # 1. 预加载OCR模型（避免首次请求卡顿）
    await asyncio.get_event_loop().run_in_executor(get_thread_pool(), get_ocr_instance)

    # 2. 初始化Milvus
    loop = asyncio.get_running_loop()
    _milvus_collection = await loop.run_in_executor(None, new_milvus_collection)

    # 3. 初始化HTTP连接池
    await get_session()

    logger.info("=== 服务启动完成，所有资源已就绪 ===")

    yield

    # Shutdown - 清理资源
    logger.info("=== 服务关闭中，清理资源 ===")
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
    if _aiohttp_session and not _aiohttp_session.closed:
        await _aiohttp_session.close()
    logger.info("=== 资源清理完成 ===")


app = FastAPI(lifespan=lifespan)


class ParseResumeRequest(BaseModel):
    resume_path: str


class LocationItem(BaseModel):
    province: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None


class SearchRequest(BaseModel):
    resume: Optional[dict] = None
    locations: Optional[List[LocationItem]] = None
    weekWorkDays: Optional[int] = None
    jobTitles: List[str] = None
    education: Optional[int] = None
    refreshTime: Optional[int] = None
    gender: Optional[int] = None
    salarySettle: Optional[str] = None
    is_open: Optional[int] = None
    pageNum: Optional[int] = None
    pageSize: Optional[int] = None


def _req_to_filters(req: SearchRequest) -> Dict[str, Any]:
    # 兼容 pydantic v1/v2
    try:
        return req.dict(exclude_none=True)  # type: ignore[attr-defined]
    except Exception:
        return req.model_dump(exclude_none=True)  # type: ignore[attr-defined]


def detect_input_type(inputs: str) -> str:
    """
    返回：
    - 'pdf'     : 单个 PDF
    - 'image'   : 单图 / 多图
    """
    ext = os.path.splitext(inputs)[1].lower()

    if ext == ".pdf":
        return "pdf"
    if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
        return "image"
    else:
        return "None"


# ============ 优化2: 所有同步IO操作放入线程池 ============

def _sync_ocr_process(img_path: str) -> str:
    """在线程池中执行的OCR处理（完全同步逻辑）"""
    # 读取为字节数组
    img_array = np.fromfile(img_path, dtype=np.uint8)
    # 解码为图像矩阵
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # OCR识别
    ocr = get_ocr_instance()
    result, _ = ocr(img)
    texts = [r[1] for r in result if r[1].strip()]
    return "\n".join(texts)


def _sync_pdf_process(pdf_path: str) -> str:
    """在线程池中执行的PDF处理（完全同步逻辑）"""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()

        # 寻找候选分割线 (x 轴 25% - 45% 区域)
        page_width = page.width
        has_text = [False] * int(page_width)
        for w in words:
            for x in range(int(w['x0']), int(w['x1'])):
                if x < len(has_text):
                    has_text[x] = True

        split_x = None
        for x in range(int(page_width * 0.25), int(page_width * 0.45)):
            if not has_text[x]:
                split_x = x
                break

        # 核心逻辑：判断是"独立双栏"还是"横向表格"
        is_true_double_column = False
        if split_x:
            left_words = [w for w in words if w['x1'] <= split_x]
            right_words = [w for w in words if w['x0'] >= split_x]

            overlap_rows = 0
            for lw in left_words:
                for rw in right_words:
                    if abs(lw['top'] - rw['top']) < 3:
                        overlap_rows += 1
                        break

            if len(left_words) > 0 and (overlap_rows / len(left_words)) < 0.3:
                is_true_double_column = True

        if is_true_double_column:
            left_box = (0, 0, split_x, page.height)
            right_box = (split_x, 0, page_width, page.height)
            content = page.crop(left_box).extract_text() + "\n" + page.crop(right_box).extract_text()
        else:
            content = page.extract_text(layout=True)

        return content


async def extract_ocr_content_async(img_path: str) -> str:
    """异步包装OCR处理"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(get_thread_pool(), _sync_ocr_process, img_path)


async def extract_text_from_pdf_async(pdf_path: str) -> str:
    """异步包装PDF处理"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(get_thread_pool(), _sync_pdf_process, pdf_path)


# ============ 优化3: 大模型API调用优化（连接池复用+重试+并发控制） ============

async def resume_text_get(resume_text: str, max_retries: int = 2) -> Dict:
    """优化后的大模型解析，带重试机制和连接池复用"""
    parse_resume_prompt = """
    你是一个简历解析助手。
    请将以下OCR文本解析为JSON，严格按照下面结构输出：
    {
      "basic_info": {
        "name": "",
        "age": 0,
        "phone": "",
        "email": "",
        "degree": "",
        "school": "",
        "major": "",
        "intent": "",
        "courses": ""
        "GPA": "",
      },
      "edu_bg": [
        {
          "duration": "",
          "school": "",
          "content": ""
        }
      ],
      "project_exp": [
        {
          "duration": "",
          "company": "",
          "content": ""
        }
      ],
      "campus_exp": [
        {
          "duration": "",
          "company": "",
          "content": ""
        }
      ],
      "award": [],
      "skills": [],
      "self": []
    }

    要求：
    1. 任何字段缺失或者对应内容为空则该字段不必加入最终结果中，类似"award": []和"self":[]这种形式也不必加入最终结果中。
    2. 日期格式如"2024.09-2026.06"。
    3. 输出必须是合法JSON，不要额外解释或注释。
    4. 如果存在GPA或绩点，像 3.8/4 这种格式输出。
    5. 如果学历存在多个，取最高的学历，如果存在，输出["大专", "本科", "硕士", "博士"]内的一个，如果没有或不符合，默认本科。
    6. 各个键顺序和结果，注意不能乱(尤其是courses字段)，请按照上面结构顺序填写。
    OCR文本如下：
    ---

    """

    data = {
        "model": "qwen-flash",
        "messages": [
            {"role": "system", "content": parse_resume_prompt},
            {"role": "user", "content": f"请分析以下简历文本：\n{resume_text}"}
        ],
        "response_format": {"type": "json_object"}
    }

    session = await get_session()

    # 重试机制
    for attempt in range(max_retries + 1):
        try:
            async with session.post(QWEN_API_URL, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    return json.loads(content)
                else:
                    error_text = await response.text()
                    logger.warning(f"API返回错误状态码: {response.status}, 内容: {error_text}")
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # 指数退避
                        logger.info(f"第{attempt + 1}次重试，等待{wait_time}秒...")
                        await asyncio.sleep(wait_time)
                    else:
                        return {"error": f"API返回错误: {response.status}, {error_text}"}

        except asyncio.TimeoutError:
            logger.warning(f"请求超时 (尝试 {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
            else:
                return {"error": "请求超时，请稍后重试"}
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            if attempt < max_retries:
                await asyncio.sleep(1)
            else:
                return {"error": str(e)}

    return {"error": "未知错误"}


# ============ 优化4: 主流程增加并发控制和异步化 ============

async def parse_resume(inputs: str) -> Dict:
    """解析文件 - 完全异步化版本"""
    input_type = detect_input_type(inputs)
    logger.info(f"简历文件类型：{input_type}")

    # 使用信号量限制并发，防止资源耗尽
    async with get_semaphore():
        start_time = time.time()

        try:
            # ================= 图片简历 =================
            if input_type == "image":
                # 异步OCR（在线程池中执行）
                resume_text = await extract_ocr_content_async(inputs)
                ocr_time = time.time()
                logger.info(f"图片OCR完成，耗时: {ocr_time - start_time:.2f}s")

                # 异步大模型解析
                qwen_result = await resume_text_get(resume_text)
                logger.info(f"大模型解析完成，总耗时: {time.time() - start_time:.2f}s")
                return qwen_result

            # ================= PDF 简历 =================
            elif input_type == "pdf":
                # 异步PDF处理（在线程池中执行）
                text = await extract_text_from_pdf_async(inputs)
                pdf_time = time.time()
                logger.info(f"PDF解析完成，耗时: {pdf_time - start_time:.2f}s")

                # 异步大模型解析
                qwen_result = await resume_text_get(text)
                logger.info(f"大模型解析完成，总耗时: {time.time() - start_time:.2f}s")
                return qwen_result

            else:
                raise HTTPException(status_code=400, detail="简历格式不正确")

        except Exception as e:
            logger.error(f"解析简历失败: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")


# ============ 优化5: 文件上传接口使用异步流式处理 ============

async def async_copy_file(src, dst_path: str, chunk_size: int = 8192):
    """异步方式复制文件，避免阻塞"""
    loop = asyncio.get_event_loop()

    def _sync_copy():
        with open(dst_path, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)

    await loop.run_in_executor(None, _sync_copy)


@app.post("/api/v1/resume/parse")
async def parse(req: ParseResumeRequest) -> Any:
    """通过路径解析文件"""
    logger.info('开始解析简历（路径方式）...')
    resume_path = _req_to_filters(req).get("resume_path")
    if not resume_path:
        logger.error('未提供简历路径...')
        raise HTTPException(status_code=400, detail="未提供简历路径")

    if not os.path.exists(resume_path):
        return {"success": False, "msg": f"文件路径不存在: {resume_path}"}

    resumes_json = await parse_resume(resume_path)
    return {
        "success": True,
        "resume": resumes_json
    }


@app.post("/api/v1/resume/parse_file")
async def parse_upload(file: UploadFile = File(...)) -> Any:
    """通过文件上传解析简历文件 - 优化版本"""
    logger.info(f'开始解析上传的简历: {file.filename}')

    if not file:
        logger.error('未上传文件...')
        raise HTTPException(status_code=400, detail="未上传文件")

    file_ext = os.path.splitext(file.filename)[1].lower()
    tmp_path = None

    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp_path = tmp.name
            # 异步写入文件内容，避免阻塞事件循环
            await async_copy_file(file.file, tmp_path)

        logger.info(f"文件已保存到临时路径: {tmp_path}")

        # 解析简历
        resumes_json = await parse_resume(tmp_path)

        return {
            "success": True,
            "filename": file.filename,
            "resume": resumes_json
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理上传文件失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")

    finally:
        # 清理资源
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"临时文件已清理: {tmp_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {tmp_path}, 错误: {e}")


# ============ 优化6: 添加健康检查接口 ============

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "platform": platform.system(),
        "ocr_ready": _ocr_instance is not None,
        "thread_pool_workers": _thread_pool._max_workers if _thread_pool else 0,
        "semaphore_limit": 3
    }


if __name__ == '__main__':
    import uvicorn  # type: ignore

    host = "0.0.0.0"
    port = 8120

    print(f"Starting optimized server at http://{host}:{port}")
    print(f"Platform: {platform.system()}")

    # Windows下使用标准uvicorn配置（不使用uvloop）
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        workers=1,  # Windows下建议workers=1，多进程使用Gunicorn或手动启动多个实例
        loop="asyncio",  # 显式指定使用标准asyncio（Windows下自动使用ProactorEventLoop）
        http="h11"  # 使用h11（纯Python，Windows兼容性好）
    )
