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
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, Function, utility  # type: ignore
from rapidocr_onnxruntime import RapidOCR
import logging
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import platform
import aiomysql
from urllib.parse import unquote
from job_recommend_campus import SearchRequestSXXZ, search_sxxz_service_optimized
from job_recommend_part_time import SearchPartRequest, search_parttime_service_optimized
from common import load_local_embeddings_cache

# 数据库连接配置
config = {
    "host": "192.168.0.218",
    "port": 3306,
    "user": "root",
    "password": "root123",
    "database": "pu_recruitment",  # 重要：替换为实际数据库名
    "charset": "utf8mb4"
    # "cursorclass": pymysql.cursors.DictCursor
}
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


async def get_db_pool():
    """获取数据库连接池"""
    global _db_pool
    if '_db_pool' not in globals() or _db_pool is None:
        _db_pool = await aiomysql.create_pool(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            db=config["database"],
            charset=config["charset"],
            autocommit=True,
            minsize=1,
            maxsize=10
        )
    return _db_pool


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
    global _milvus_collection, _db_pool  # 添加 _db_pool 到 global

    logger.info("=== 服务启动中，预初始化资源 ===")

    # 1. 预加载OCR模型
    await asyncio.get_event_loop().run_in_executor(get_thread_pool(), get_ocr_instance)

    # 2. 初始化Milvus
    loop = asyncio.get_running_loop()
    _milvus_collection = await loop.run_in_executor(None, new_milvus_collection)

    # 2.1 初始化数据库连接池（确保 _db_pool 被设置）
    _db_pool = await get_db_pool()

    # 3. 初始化HTTP连接池
    await get_session()

    # 4. 【关键修复】共享资源给 campus 和 part_time 模块
    from job_recommend_campus import _state as campus_state
    from job_recommend_part_time import _state as part_time_state
    from common import AsyncEmbeddingClient, AsyncRerankClient, IntelligentReranker, OnlineIntelligentReranker
    from config import EMBEDDING_CONCURRENCY, MILVUS_CONCURRENCY

    # 获取当前 session 用于创建客户端
    session = await get_session()

    # 共享给 campus 模块
    if campus_state.db_pool is None:
        campus_state.db_pool = _db_pool
        campus_state.milvus_collection = _milvus_collection
        campus_state.http_session = session
        campus_state.embedding_client = AsyncEmbeddingClient(session, concurrency=EMBEDDING_CONCURRENCY)
        campus_state.rerank_client = AsyncRerankClient(session)
        campus_state.intelligent_reranker = IntelligentReranker(campus_state.rerank_client)
        campus_state.online_intelligent_reranker = OnlineIntelligentReranker(campus_state.rerank_client)
        campus_state._sem_milvus = asyncio.Semaphore(MILVUS_CONCURRENCY)
        campus_state.local_emb_cache = load_local_embeddings_cache()
        logger.info("Campus 服务状态已初始化")

    # 共享给 part_time 模块
    if part_time_state.db_pool is None:
        part_time_state.db_pool = _db_pool
        part_time_state.milvus_collection = _milvus_collection
        part_time_state.http_session = session
        part_time_state.embedding_client = AsyncEmbeddingClient(session, concurrency=EMBEDDING_CONCURRENCY)
        part_time_state.rerank_client = AsyncRerankClient(session)
        part_time_state.intelligent_reranker = IntelligentReranker(part_time_state.rerank_client)
        part_time_state.online_intelligent_reranker = OnlineIntelligentReranker(part_time_state.rerank_client)
        part_time_state._sem_milvus = asyncio.Semaphore(MILVUS_CONCURRENCY)
        part_time_state.local_emb_cache = load_local_embeddings_cache()
        logger.info("Part-time 服务状态已初始化")

    logger.info("=== 服务启动完成，所有资源已就绪 ===")

    yield

    # Shutdown - 清理资源
    logger.info("=== 服务关闭中，清理资源 ===")

    # 清理 campus 和 part_time 的 session（但不关闭 db_pool，因为与本模块共享）
    if campus_state.http_session and not campus_state.http_session.closed:
        await campus_state.http_session.close()
    if part_time_state.http_session and not part_time_state.http_session.closed:
        await part_time_state.http_session.close()

    if _thread_pool:
        _thread_pool.shutdown(wait=True)
    if _aiohttp_session and not _aiohttp_session.closed:
        await _aiohttp_session.close()
    if _db_pool:
        _db_pool.close()
        await _db_pool.wait_closed()

    logger.info("=== 资源清理完成 ===")


app = FastAPI(lifespan=lifespan)


class ParseResumeRequest(BaseModel):
    resume_id: int
    user_id: int
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
    - 必须提取"职位名称"
    - 多岗位用数组形式，但主岗位排第一

    禁止：
    - 段落标题
    - 行业名称
    - 模糊方向
    - 发展规划

    无法确定 → intent=[]

    ==============================

    二、work_city（意向城市提取）
    ==============================

    【提取规则】
    - 必须查找明确的"意向城市"、"期望城市"、"工作地点"、"目标城市"、"工作城市"等字段
    - 只提取城市级别（不要省、区/县），使用标准城市名称
    - 例如：上海→上海市，杭州→杭州市，澳门→澳门特别行政区，北京→北京市
    - 如果简历中没有明确提及意向城市、期望城市或工作地点，work_city必须返回空字符串""，禁止根据籍贯、家庭住址、学校所在地等信息臆测

    【输出格式】
    - 涉及多个城市则以","分隔，如"上海市,杭州市,深圳市"
    - 没有明确提及则返回""

    ==============================

    三、基本信息
    ==============================

    name：提取姓名，没有则填""

    age：
    优先年龄；否则根据出生年份计算整数

    degree：
    只保留最高学历

    phone/email：
    必须格式正确，否则填 ""

    ==============================

    四、经历（最多3段）
    ==============================

    保留最近经历
    必须包含：
    - 技术栈
    - 职责关键词
    - 成果关键词

    删除空话

    时间格式：
    YYYY.MM-YYYY.MM
    至今保留"至今"

    ==============================

    五、技能（用于匹配）
    ==============================

    拆分为原子技能：
    错误：熟悉Java开发
    正确：Java, SpringBoot, MySQL

    必须包含：
    语言 / 框架 / 数据库 / 工具 / 平台

    ==============================

    六、输出格式（严格）
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
        "work_city": "",
        "intent": []
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

    简历文本如下：
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


# ============ 新增：简历类型判断大模型调用 ============

RESUME_TYPE_PROMPT = """
你是一个文档分类专家。请判断以下文本是否属于简历，如果是简历，进一步判断其求职类型。

【分类标准】
1. 非简历文档（输出3）：
   - 完全不是简历格式（如小说、新闻、论文、合同、说明书等）
   - 没有个人信息（姓名、联系方式）
   - 没有求职意向或工作经历/教育经历
   - 内容明显与求职无关

2. 实习/校招简历（输出1）：
   - 明确出现"实习"、"校招"、"校园招聘"、"应届生"、"毕业生"等关键词
   - 技术岗位（工程师、开发、算法等）
   - 工作年限>1年或项目制工作经历
   - 有明确的教育背景和工作/项目经历
   - 不确定或模糊的情况（默认此类）

3. 兼职简历（输出2）必须同时满足：
   - 明确出现"兼职"关键词
   - 岗位为服务型（服务员/配送/促销/家教等）或明确提到"小时工/临时工/日结/周结/灵活时间"

【输出要求】
- 仅输出单个数字：1、2 或 3
- 3 = 非简历文档
- 1 = 实习/校招简历（默认）
- 2 = 兼职简历（必须明确满足条件）
- 不要输出任何解释、标点或其他文字

【重要】宁可将不确定的判为1，也不要轻易判为2或3。

文档内容：
---
{text}
---
"""


async def classify_resume_type(resume_text: str, max_retries: int = 2) -> int:
    """
    使用大模型判断简历类型
    返回: 1=实习/校招, 2=兼职
    """
    # 截取前3000字作为判断依据（足够判断类型，节省token）
    text_sample = resume_text[:3000] if len(resume_text) > 3000 else resume_text

    data = {
        "model": "qwen-turbo",  # 使用轻量级模型，快速判断
        "temperature": 0.1,
        "max_tokens": 10,  # 只需要输出一个数字
        "messages": [
            {"role": "system", "content": "你是一个简历分类助手，只输出数字1或2。"},
            {"role": "user", "content": RESUME_TYPE_PROMPT.format(text=text_sample)}
        ]
    }

    session = await get_session()

    for attempt in range(max_retries + 1):
        try:
            async with session.post(QWEN_API_URL, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

                    # 提取数字
                    import re
                    numbers = re.findall(r'\b[123]\b', content)  # 只匹配1、2、3

                    if numbers:
                        doc_type = int(numbers[0])
                        type_map = {1: "实习/校招", 2: "兼职", 3: "非简历"}
                        logger.info(f"文档类型判断结果: {doc_type} ({type_map.get(doc_type, '未知')})")
                        return doc_type

                    # 如果没有匹配到有效数字，保守默认为实习/校招(1)
                    logger.warning(f"无法从模型输出中提取有效类型，输出内容: '{content}'，默认返回1(实习/校招)")
                    return 3

                else:
                    error_text = await response.text()
                    logger.warning(f"类型判断API错误: {response.status}")
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        # API失败时保守默认
                        return 3

        except Exception as e:
            logger.error(f"类型判断异常: {str(e)}")
            if attempt < max_retries:
                await asyncio.sleep(1)
            else:
                return 1  # 默认实习/校招

    return 1


async def save_parse_result_to_db(
        resume_id: int,
        user_id: int,
        file_path: str,
        parsed_result: Dict,
        resume_type: int,
        name: str = "",  # 新增：姓名
        work_city: str = ""
) -> bool:
    """
    将解析结果保存到数据库
    先判断是否存在，存在则更新，不存在则插入
    """
    try:
        pool = await get_db_pool()
        post_address_json = json.dumps(parsed_result, ensure_ascii=False)
        # 限制字段长度
        name = (name or "")[:100]
        work_city = (work_city or "")[:100]
        # work_type映射: 2=实习/校招, 3=兼职
        work_type = 2 if resume_type == 1 else 3

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # 先查询是否存在
                await cur.execute(
                    "SELECT id FROM pr_resume_parse WHERE id = %s",
                    (resume_id,)
                )
                existing = await cur.fetchone()

                if existing:
                    # 存在，执行更新（包含name和work_city）
                    logger.info(f"简历记录已存在，执行更新: resume_id={resume_id}")
                    update_sql = """
                                        UPDATE pr_resume_parse SET
                                            user_id = %s,
                                            name = %s,
                                            resume_type = %s,
                                            is_deleted = %s,
                                            preview_address = %s,
                                            work_city = %s,
                                            update_time = NOW(),
                                            post_address = %s
                                        WHERE id = %s
                                        """
                    await cur.execute(update_sql, (
                        user_id,
                        name,  # 新增
                        resume_type,
                        0,
                        file_path[:500] if file_path else None,
                        work_city,  # 新增
                        post_address_json,
                        resume_id
                    ))
                else:
                    # 不存在，执行插入（包含name和work_city）
                    logger.info(f"简历记录不存在，执行插入: resume_id={resume_id}")
                    insert_sql = """
                                       INSERT INTO pr_resume_parse (
                                           id, user_id, name,  resume_type, is_deleted,
                                           preview_address, work_city, create_time, update_time, post_address
                                       ) VALUES (
                                           %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s
                                       )
                                       """
                    await cur.execute(insert_sql, (
                        resume_id,
                        user_id,
                        name,  # 新增
                        resume_type,
                        0,
                        file_path[:500] if file_path else None,
                        work_city,  # 新增
                        post_address_json
                    ))

                await conn.commit()

        action = "更新" if existing else "插入"
        logger.info(f"简历解析结果已{action}到数据库: resume_id={resume_id}, type={resume_type}")
        return True

    except Exception as e:
        logger.error(f"保存到数据库失败: {str(e)}", exc_info=True)
        return False


async def background_parse_and_save(
        resume_text: str,
        file_path: str,
        resume_id: int,
        user_id: int,
        resume_type: int  # 传入已判断的类型
):
    """
    后台任务：执行完整大模型解析并保存到数据库
    """
    try:
        logger.info(f"开始后台完整解析: resume_id={resume_id}, type={resume_type}")
        start_time = time.time()

        # 调用完整解析大模型
        parsed_result = await resume_text_get(resume_text)
        print('parsed_result--->', parsed_result)
        parse_time = time.time() - start_time

        logger.info(f"完整解析完成，耗时: {parse_time:.2f}s")
        # 从解析结果中提取name和work_city
        basic_info = parsed_result.get("basic_info", {}) if "error" not in parsed_result else {}
        name = basic_info.get("name", "")
        work_city = basic_info.get("work_city", "")

        if "error" in parsed_result:
            logger.error(f"完整解析失败: {parsed_result.get('error')}")
            # 即使解析失败，也保存基础信息和类型
            parsed_result = {
                "basic_info": {
                    "name": name,
                    "work_city": work_city,
                    "intent": []
                },
                "error": parsed_result.get("error")
            }

        # 保存到数据库（带上类型信息）
        save_success = await save_parse_result_to_db(
            resume_id=resume_id,
            user_id=user_id,
            file_path=file_path,
            parsed_result=parsed_result,
            resume_type=resume_type,
            name=name,  # 传递name
            work_city=work_city
        )

        if save_success:
            logger.info(f"后台处理完成: resume_id={resume_id}, 总耗时: {time.time() - start_time:.2f}s")
        else:
            logger.error(f"数据库保存失败: resume_id={resume_id}")

    except Exception as e:
        logger.error(f"后台处理异常: {str(e)}", exc_info=True)


# ============ 优化4: 主流程增加并发控制和异步化 ============

async def parse_resume(inputs: str, resume_path: str, resume_id: int, user_id: int,
                       background_tasks: BackgroundTasks):
    """解析文件 - 完全异步化版本"""
    input_type = detect_input_type(inputs)
    logger.info(f"简历文件类型：{input_type}")

    # 使用信号量限制并发，防止资源耗尽
    async with get_semaphore():
        start_time = time.time()

        try:
            # ================= 步骤1: 提取文本内容 =================
            logger.info(f"开始提取文本: resume_id={resume_id}")
            start_time = time.time()
            if input_type == "image":
                resume_text = await extract_ocr_content_async(inputs)
                logger.info(f"图片OCR完成，耗时: {time.time() - start_time:.2f}s")

            elif input_type == "pdf":
                resume_text = await extract_text_from_pdf_async(inputs)
                logger.info(f"PDF解析完成，耗时: {time.time() - start_time:.2f}s")

            else:
                # raise HTTPException(status_code=400, detail="简历格式不正确")
                return {"success": False, "message": f"简历格式不正确"}
            extract_time = time.time() - start_time
            text_length = len(resume_text)
            logger.info(f"文本提取完成: 长度={text_length}, 耗时={extract_time:.2f}s")
            # ================= 步骤2: 判断简历类型 =================
            logger.info(f"开始判断简历类型: resume_id={resume_id}")
            type_start = time.time()
            resume_type = await classify_resume_type(resume_text)
            type_time = time.time() - type_start

            type_label = "实习/校招" if resume_type == 1 else "兼职"
            logger.info(f"简历类型判断完成: {type_label}({resume_type}), 耗时={type_time:.2f}s")
            # ========== 步骤3：触发后台完整解析任务 ==========
            if (resume_type == 1 or resume_type == 2):
                background_tasks.add_task(
                    background_parse_and_save,
                    resume_text=resume_text,
                    file_path=resume_path,
                    resume_id=resume_id,
                    user_id=user_id,
                    resume_type=resume_type  # 传递判断好的类型
                )
                logger.info(f"已触发后台完整解析任务: resume_id={resume_id}")
                return True
            else:
                logger.info(f"简历类型为非实习/校招，不触发后台完整解析任务: resume_id={resume_id}")
                return False
        except Exception as e:
            logger.error(f"解析简历失败: {str(e)}", exc_info=True)
            # raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")
            return {"success": False, "message": f"解析失败: {str(e)}"}


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
async def parse(req: ParseResumeRequest, background_tasks: BackgroundTasks) -> Any:
    """通过路径或URL解析文件"""
    logger.info('开始解析简历...')

    req = req.model_dump(exclude_none=True)
    resume_path = req.get("resume_path")
    resume_id = req.get("resume_id")
    user_id = req.get("user_id")
    if not resume_path:
        logger.error('未提供简历路径...')
        # raise HTTPException(status_code=400, detail="未提供简历路径")
        return {"success": False, "message": "未提供简历路径"}
    temp_dir = None

    try:
        # 判断是URL还是本地路径
        if resume_path.startswith(('http://', 'https://')):
            # 是HTTP URL，需要下载
            logger.info(f'检测到远程URL，开始下载: {resume_path}')

            # URL解码（处理中文文件名）
            decoded_url = unquote(resume_path)

            # 创建临时目录
            temp_dir = tempfile.mkdtemp()

            # 从URL中提取文件名
            file_name = decoded_url.split('/')[-1]
            if not file_name:
                file_name = "resume.pdf"

            local_file_path = os.path.join(temp_dir, file_name)

            # 异步下载文件
            async with aiohttp.ClientSession() as session:
                async with session.get(decoded_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        # raise HTTPException(
                        #     status_code=400,
                        #     detail=f"无法下载文件，HTTP状态码: {response.status}"
                        # )
                        return {"success": False, "msg": f"无法下载文件，HTTP状态码: {response.status}"}

                    # 保存到临时文件
                    with open(local_file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

            logger.info(f'文件下载完成: {local_file_path}')
            parse_path = local_file_path

        else:
            # 是本地路径
            if not os.path.exists(resume_path):
                return {"success": False, "msg": f"文件路径不存在: {resume_path}"}
            parse_path = resume_path

        # 解析简历
        flag = await parse_resume(parse_path, resume_path, resume_id, user_id, background_tasks)
        msg = "简历解析成功" if flag else "简历解析失败，该文件内容不包含简历信息"
        return {
            "success": flag,
            "msg": msg
        }

    except aiohttp.ClientError as e:
        logger.error(f'下载文件失败: {str(e)}')
        # raise HTTPException(status_code=400, detail=f"下载文件失败: {str(e)}")
        return {"success": False, "msg": f"文件路径不存在: {resume_path}"}
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v1/jobs/list_recommend_Resume")
async def search_sxxz_with_resume(req: SearchRequestSXXZ):
    logger.info(f"进入实习校招岗位推荐系统(优化版)...")
    filters = _req_to_filters(req)
    if "is_open" not in filters or filters.get("is_open") is None:
        filters["is_open"] = 1
    print('filters---->', filters)
    return await search_sxxz_service_optimized(filters)


@app.post("/api/v1/part-time-jobs/search_with_Resume")
async def search_parttime_with_resume(req: SearchPartRequest):
    filters = _req_to_filters(req)
    if "is_open" not in filters or filters.get("is_open") is None:
        filters["is_open"] = 1
    filters["workNature"] = 3
    return await search_parttime_service_optimized(filters)


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
