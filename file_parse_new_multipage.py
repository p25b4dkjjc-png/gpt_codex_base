import asyncio
import json
import os
import re
import uuid
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
_llm_parse_tasks: Dict[str, Dict[str, Any]] = {}


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
    """在线程池中执行的PDF处理（支持多页解析）"""
    all_pages_content = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"PDF总页数: {total_pages}")

        for page_num, page in enumerate(pdf.pages, 1):
            try:
                # 提取当前页面的文本
                page_content = _extract_single_page(page, page_num, total_pages)
                if page_content and page_content.strip():
                    all_pages_content.append(page_content)
                    logger.debug(f"第{page_num}页解析完成，内容长度: {len(page_content)}")
            except Exception as e:
                logger.warning(f"第{page_num}页解析失败: {e}")
                continue

    # 合并所有页面内容
    full_content = "\n\n".join(all_pages_content)
    logger.info(f"PDF解析完成，共{len(all_pages_content)}页有内容，总长度: {len(full_content)}")

    return full_content


def _extract_single_page(page, page_num: int, total_pages: int) -> str:
    """提取单页内容 - 保留原有的双栏检测逻辑"""
    words = page.extract_words()

    if not words:
        logger.debug(f"第{page_num}页无文本内容")
        return ""

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

    # 提取文本
    if is_true_double_column:
        left_box = (0, 0, split_x, page.height)
        right_box = (split_x, 0, page_width, page.height)
        left_text = page.crop(left_box).extract_text() or ""
        right_text = page.crop(right_box).extract_text() or ""
        content = left_text + "\n" + right_text
    else:
        content = page.extract_text(layout=True) or ""

    # 添加页码标记（便于调试）
    # if content.strip():
    #     content = f"<!-- Page {page_num}/{total_pages} -->\n{content}"

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
你是【岗位匹配导向】的简历结构化解析助手。
目标：提取可用于岗位匹配的高质量结构化数据。
仅输出JSON，不输出解释。

==============================
一、intent（最高优先级字段）
==============================
【步骤1：明确标签提取（最高优先级）】
查找以下字段：

求职意向
期望岗位
应聘岗位
目标职位
应聘方向
Job Objective
Target Position
Desired Position

要求：
- 必须提取“职位名称”
- 多岗位用数组形式，但主岗位排第一

禁止：
- 段落标题
- 行业名称
- 模糊方向
- 发展规划

无法确定 → intent=""

==============================
二、基本信息
==============================

age：
优先年龄；否则根据出生年份计算整数

degree：
只保留最高学历

phone/email：
必须格式正确，否则填 ""

==============================
三、经历（最多3段）
==============================

保留最近经历
必须包含：
- 技术栈
- 职责关键词
- 成果关键词

删除空话

时间格式：
YYYY.MM-YYYY.MM
至今保留“至今”

==============================
四、技能（用于匹配）
==============================

拆分为原子技能：
错误：熟悉Java开发
正确：Java, SpringBoot, MySQL

必须包含：
语言 / 框架 / 数据库 / 工具 / 平台

==============================
五、输出格式（严格）
==============================

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
    "courses": "",
    "GPA": ""
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
  "award": [],
  "skills": []
}

字段缺失填 "" 或 []
只输出JSON

--------------------------------

OCR文本如下：
---


"""
    MAX_INPUT_LENGTH = 12000  # 防止超长简历拖慢模型
    if len(resume_text) > MAX_INPUT_LENGTH:
        resume_text = resume_text[:MAX_INPUT_LENGTH]
    data = {
        "model": "qwen-plus",
        "temperature": 0.1,  # 降低随机性
        "top_p": 0.8,
        "max_tokens": 2000,  # 限制输出长度
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": parse_resume_prompt},
            {"role": "user", "content": resume_text}
        ]
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


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip("：:|·•- \t")


def _split_resume_lines(resume_text: str) -> List[str]:
    return [_normalize_line(line) for line in resume_text.splitlines() if _normalize_line(line)]


def _extract_intent(lines: List[str]) -> str:
    """基于规则提取求职意向（行业无关，兼容中英文简历）。"""
    heading_pattern = re.compile(
        r"^(求职意向|求职目标|应聘岗位|目标职位|期望岗位|期望职位|应聘方向|意向岗位|"
        r"job\s*objective|target\s*position|desired\s*position|career\s*objective)",
        re.IGNORECASE,
    )

    for idx, line in enumerate(lines):
        if heading_pattern.search(line):
            # 先尝试同行冒号后内容
            if "：" in line or ":" in line:
                right = _normalize_line(re.split(r"[：:]", line, maxsplit=1)[1])
                if right and len(right) <= 40:
                    return right
            # 再尝试下一行
            if idx + 1 < len(lines):
                nxt = lines[idx + 1]
                if len(nxt) <= 40 and not re.search(r"(教育|项目|工作|技能|证书|经历)", nxt):
                    return nxt

    # 全文兜底：提取“应聘/期望/目标 + 岗位名称”
    fallback_pattern = re.compile(
        r"(?:应聘|期望|目标|意向)\s*(?:岗位|职位)?\s*[：:]?\s*([\u4e00-\u9fa5A-Za-z0-9+/#\-· ]{2,30})"
    )
    for line in lines:
        match = fallback_pattern.search(line)
        if match:
            return _normalize_line(match.group(1))

    return ""


def _extract_skills(lines: List[str]) -> List[str]:
    """规则提取专业技能，支持通用行业关键词，不绑定某一垂直领域。"""
    section_keywords = re.compile(r"(专业技能|技能特长|技能清单|核心技能|职业技能|skills?)", re.IGNORECASE)
    stop_words = {
        "熟悉", "掌握", "了解", "精通", "具备", "能够", "使用", "擅长", "技能", "专业", "核心", "熟练",
        "good", "proficient", "familiar", "skilled", "with", "and", "or"
    }
    skill_pattern = re.compile(
        r"\b(?:[A-Za-z][A-Za-z0-9+.#/_-]{1,20}|[\u4e00-\u9fa5]{2,12})\b"
    )

    candidates: List[str] = []
    for idx, line in enumerate(lines):
        if section_keywords.search(line):
            window = lines[idx:min(idx + 12, len(lines))]
            candidates.extend(window)

    if not candidates:
        candidates = lines

    parsed: List[str] = []
    for line in candidates:
        for token in skill_pattern.findall(line):
            token = token.strip(".,;:|()[]{}")
            if not token:
                continue
            lower_token = token.lower()
            if lower_token in stop_words:
                continue
            if len(token) <= 1:
                continue
            if re.fullmatch(r"\d+[年月日]?", token):
                continue
            parsed.append(token)

    # 去重并保序，优先保留高信息量词
    seen = set()
    skills = []
    for item in parsed:
        key = item.lower()
        if key not in seen and len(item) <= 30:
            seen.add(key)
            skills.append(item)

    return skills[:30]


def _extract_experience_by_rule(lines: List[str]) -> List[Dict[str, str]]:
    """提取项目/实习/兼职经历，统一输出便于岗位快速匹配。"""
    block_patterns = [
        re.compile(r"项目经历|项目经验|project\s*experience", re.IGNORECASE),
        re.compile(r"实习经历|实习经验|intern(ship)?\s*experience", re.IGNORECASE),
        re.compile(r"兼职经历|兼职经验|part\s*-?time\s*experience", re.IGNORECASE),
    ]
    date_pattern = re.compile(
        r"((?:19|20)\d{2}[./年-](?:0?[1-9]|1[0-2])(?:月)?\s*[-~至]\s*(?:(?:19|20)\d{2}[./年-](?:0?[1-9]|1[0-2])(?:月)?|至今|Now|Present))",
        re.IGNORECASE,
    )

    experiences: List[Dict[str, str]] = []
    for idx, line in enumerate(lines):
        if not any(p.search(line) for p in block_patterns):
            continue

        section_lines = lines[idx + 1:min(idx + 28, len(lines))]
        current = {"type": _normalize_line(line), "duration": "", "title": "", "content": ""}

        for sec_line in section_lines:
            if re.search(r"(教育|技能|证书|自我评价|校园活动|获奖|语言能力)", sec_line):
                break
            if any(p.search(sec_line) for p in block_patterns):
                break

            if not current["duration"]:
                matched = date_pattern.search(sec_line)
                if matched:
                    current["duration"] = matched.group(1)
                    continue

            if not current["title"] and 2 <= len(sec_line) <= 50:
                current["title"] = sec_line
                continue

            if len(sec_line) > 4:
                current["content"] += (sec_line + "；")

        if current["title"] or current["content"]:
            current["content"] = current["content"].strip("；")
            experiences.append(current)

    return experiences[:6]


def extract_resume_fast_fields(resume_text: str) -> Dict[str, Any]:
    """主线程规则提取：用于快速岗位匹配。"""
    lines = _split_resume_lines(resume_text)
    intent = _extract_intent(lines)
    skills = _extract_skills(lines)
    exp = _extract_experience_by_rule(lines)
    return {
        "intent": intent,
        "skills": skills,
        "project_or_internship_or_parttime_exp": exp,
    }


async def _run_llm_parse_task(task_id: str, resume_text: str) -> None:
    """后台异步执行大模型结构化解析，结果可用于后续入库。"""
    try:
        _llm_parse_tasks[task_id]["status"] = "running"
        _llm_parse_tasks[task_id]["result"] = await resume_text_get(resume_text)
        _llm_parse_tasks[task_id]["status"] = "done"
    except Exception as e:
        _llm_parse_tasks[task_id]["status"] = "failed"
        _llm_parse_tasks[task_id]["result"] = {"error": str(e)}


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

                # 主线程规则提取（用于快速匹配岗位）
                fast_fields = extract_resume_fast_fields(resume_text)

                # 后台异步大模型提取（用于后续入库）
                task_id = str(uuid.uuid4())
                _llm_parse_tasks[task_id] = {"status": "queued", "result": None}
                asyncio.create_task(_run_llm_parse_task(task_id, resume_text))

                logger.info(f"规则提取完成，总耗时: {time.time() - start_time:.2f}s")
                return {
                    "fast_extract": fast_fields,
                    "llm_task_id": task_id,
                    "llm_task_status": "queued",
                }

            # ================= PDF 简历 =================
            elif input_type == "pdf":
                # 异步PDF处理（在线程池中执行）
                text = await extract_text_from_pdf_async(inputs)
                pdf_time = time.time()
                logger.info(f"PDF解析完成，耗时: {pdf_time - start_time:.2f}s")

                # 主线程规则提取（用于快速匹配岗位）
                fast_fields = extract_resume_fast_fields(text)

                # 后台异步大模型提取（用于后续入库）
                task_id = str(uuid.uuid4())
                _llm_parse_tasks[task_id] = {"status": "queued", "result": None}
                asyncio.create_task(_run_llm_parse_task(task_id, text))

                logger.info(f"规则提取完成，总耗时: {time.time() - start_time:.2f}s")
                return {
                    "fast_extract": fast_fields,
                    "llm_task_id": task_id,
                    "llm_task_status": "queued",
                }

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


@app.get("/api/v1/resume/parse_task/{task_id}")
async def get_parse_task(task_id: str) -> Any:
    """查询后台大模型结构化解析任务状态。"""
    task = _llm_parse_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "task_id": task_id, **task}


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
