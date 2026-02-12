import asyncio
import json
import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, Tuple, Optional, Union, List
import tempfile
import shutil
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import platform

import dashscope
import numpy as np
import aiohttp
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, Function, utility

# ----------------------------- 配置区 -----------------------------
MILVUS_HOST = "192.168.0.2"
MILVUS_PORT = "19530"
COLLECTION_NAME = "job_recommendations"
EMBEDDING_DIM = 1024
VECTOR_DIM = EMBEDDING_DIM

DASHSCOPE_API_KEY = "sk-b8855f2599ea4111a20e3b621713e97b"
QWEN_API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
EMBEDDING_API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'
RERANK_API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'

if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("<"):
    raise RuntimeError("未提供 DashScope API Key")
dashscope.api_key = DASHSCOPE_API_KEY

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Windows优化
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ----------------------------- 全局资源 -----------------------------
_thread_pool: Optional[ThreadPoolExecutor] = None
_aiohttp_session: Optional[aiohttp.ClientSession] = None
_milvus_collection: Optional[Collection] = None
_embedding_cache: Dict[str, List[float]] = {}
_local_emb_cache: Optional[Dict[str, List[float]]] = None
# 并发控制
_sem_embedding = asyncio.Semaphore(5)  # 向量化并发限制
_sem_milvus = asyncio.Semaphore(3)  # Milvus查询并发限制


def get_thread_pool() -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        max_workers = min(16, (os.cpu_count() or 1) * 2)
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool


async def get_session() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        _aiohttp_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
        )
    return _aiohttp_session


def get_local_embeddings_cache() -> Dict[str, List[float]]:
    global _local_emb_cache
    if _local_emb_cache is None:
        current_dir = os.path.dirname(__file__)
        cache_path = os.path.join(current_dir or ".", "hopejobs_qwen_v4_text_embeddings.json")
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    data = obj.get("data") if isinstance(obj, dict) else obj
                    _local_emb_cache = data if isinstance(data, dict) else {}
            else:
                _local_emb_cache = {}
        except Exception:
            _local_emb_cache = {}
    return _local_emb_cache


def new_milvus_collection() -> Collection:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)
    try:
        col.load()
    except Exception:
        pass
    return col


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _milvus_collection
    logger.info("=== 服务启动，预初始化资源 ===")

    # 预加载本地缓存
    get_local_embeddings_cache()

    # 预连接Milvus
    loop = asyncio.get_running_loop()
    _milvus_collection = await loop.run_in_executor(None, new_milvus_collection)

    # 预初始化HTTP session
    await get_session()

    logger.info("=== 服务启动完成 ===")
    yield

    if _thread_pool:
        _thread_pool.shutdown(wait=True)
    if _aiohttp_session and not _aiohttp_session.closed:
        await _aiohttp_session.close()


app = FastAPI(lifespan=lifespan)


# ----------------------------- Pydantic模型 -----------------------------
class LocationItem(BaseModel):
    province: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None


class SearchRequestSXXZ(BaseModel):
    resume: Optional[dict] = None
    workNature: int
    locations: Optional[List[LocationItem]] = None
    salaryRanges: Optional[List[int]] = None
    education: Optional[int] = None
    weekWorkDays: Optional[int] = None
    internshipMonths: Optional[List[int]] = None
    is_remote: Optional[int] = None
    refreshTime: Optional[int] = None
    gender: Optional[int] = None
    salarySettle: Optional[str] = None
    is_open: Optional[int] = None
    pageNum: Optional[int] = None
    pageSize: Optional[int] = None


# ----------------------------- 异步向量化客户端 -----------------------------
class AsyncEmbeddingClient:
    """异步向量化客户端，支持并发和缓存"""

    @staticmethod
    async def embed_texts(texts: List[str], dimension: int = VECTOR_DIM) -> List[List[float]]:
        """异步批量向量化，带并发控制"""
        if not texts:
            return []

        # 去重，减少API调用
        unique_texts = list(set(t.strip() for t in texts if t and t.strip()))
        if not unique_texts:
            return [[] for _ in texts]

        # 检查缓存
        cache = get_local_embeddings_cache()
        emb_cache = _embedding_cache
        results_map, texts_to_embed = {}, []

        for text in unique_texts:
            if text in emb_cache:
                results_map[text] = emb_cache[text]
            elif text in cache:
                results_map[text] = cache[text]
                emb_cache[text] = cache[text]  # 加入内存缓存
            else:
                texts_to_embed.append(text)

        # 并发调用API（带限流）
        if texts_to_embed:
            async with _sem_embedding:
                session = await get_session()
                embeddings = await asyncio.gather(
                    *(AsyncEmbeddingClient._embed_single(session, text, dimension) for text in texts_to_embed),
                    return_exceptions=True,
                )

                for text, emb in zip(texts_to_embed, embeddings):
                    if isinstance(emb, list) and len(emb) == dimension:
                        results_map[text] = emb
                        if any(emb):
                            emb_cache[text] = emb  # 缓存结果

        # 按原始顺序返回
        return [results_map.get(t.strip(), []) for t in texts]

    @staticmethod
    async def _embed_single(session: aiohttp.ClientSession, text: str, dimension: int) -> List[float]:
        """单个文本向量化"""
        payload = {
            "model": "text-embedding-v4",
            "input": {"texts": [text]},
            "parameters": {"dimension": dimension, "output_type": "dense"}
        }

        for attempt in range(2):  # 重试机制
            try:
                async with session.post(EMBEDDING_API_URL, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("output") and data["output"].get("embeddings"):
                            emb = data["output"]["embeddings"][0]["embedding"]
                            return [float(x) for x in emb]
                    else:
                        error = await resp.text()
                        logger.warning(f"Embedding API error: {resp.status}, {error}")
                        if attempt == 0:
                            await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Embedding request failed: {e}")
                if attempt == 0:
                    await asyncio.sleep(0.5)

        # 失败时返回零向量
        return [0.0] * dimension


class AsyncRerankClient:
    """异步重排客户端 - 修复版"""

    @staticmethod
    async def rerank(query: str, documents: List[str], top_k: int = 100) -> List[Dict]:
        """异步重排 - 修复字段名问题"""
        if not documents or not query:
            return []

        payload = {
            "model": "qwen3-rerank",
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {"top_n": min(top_k, len(documents)), "return_documents": False}
        }

        session = await get_session()
        try:
            async with session.post(RERANK_API_URL, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("output", {}).get("results", ())

                    # ----------------------------- 修复2: 正确的字段名 ----------------------------
                    # DashScope gte-rerank-v2 返回的是 relevance_score 而不是 score
                    rerank_results = []
                    append = rerank_results.append
                    for r in results:
                        # 安全获取字段，兼容不同版本的API
                        index = r.get("index")
                        # 优先使用 relevance_score，回退到 score
                        score = r.get("relevance_score") or r.get("score") or 0.0

                        if index is not None:
                            append({
                                "index": int(index),
                                "score": float(score)
                            })

                    logger.info(f"Rerank成功: 返回{len(rerank_results)}个结果")
                    return rerank_results
                else:
                    error = await resp.text()
                    logger.error(f"Rerank API error: {resp.status}, {error}")
                    return []
        except Exception as e:
            logger.error(f"Rerank request failed: {e}", exc_info=True)
            return []


# ----------------------------- 工具函数 -----------------------------
def _normalize_page(page_num: Optional[int], page_size: Optional[int]) -> Tuple[int, int, int]:
    pn = 1 if page_num is None or not isinstance(page_num, int) or page_num < 1 else page_num
    if page_size is None or not isinstance(page_size, int):
        ps = 10
    else:
        ps = 10 if page_size < 1 else (100 if page_size > 100 else page_size)
    return pn, ps, (pn - 1) * ps


def cosine_sim(a, b):
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def rrf_fusion(rank_maps: dict, k: int = 60):
    scores = defaultdict(float)
    for results in rank_maps.values():
        for rank, job_id in enumerate(results):
            scores[job_id] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [job_id for job_id, _ in ranked], [score for _, score in ranked]


def _and_expr(exprs) -> Optional[str]:
    exprs = [e for e in exprs if e]
    return " and ".join(f"({e})" for e in exprs) if exprs else None


def _dedupe_keep_order(items):
    return list(dict.fromkeys(items))


# ----------------------------- 修复1: Pydantic 2.0 兼容性 ----------------------------
def _req_to_filters(req) -> Dict[str, Any]:
    """兼容 Pydantic v1/v2，优先使用 model_dump (v2)"""
    try:
        # Pydantic v2
        return req.model_dump(exclude_none=True)
    except AttributeError:
        # Pydantic v1 回退
        try:
            return req.dict(exclude_none=True)
        except Exception as e:
            logger.error(f"Failed to convert request to dict: {e}")
            raise


# 实习校招专用过滤条件
def _sanitize_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    return None


def _build_salary_ranges_filter_milvus_sxxz(salary_ranges: Optional[List[int]]) -> Optional[str]:
    if not salary_ranges or 0 in salary_ranges:
        return None
    range_mapping = {
        1: (0, 3000), 2: (3000, 5000), 3: (5000, 10000),
        4: (10000, 15000), 5: (15000, 20000), 6: (20000, 30000), 7: (30000, None)
    }
    range_exprs = []
    for r in salary_ranges:
        if r in range_mapping:
            min_val, max_val = range_mapping[r]
            if max_val is None:
                range_exprs.append(f"salaryMax >= {min_val}")
            else:
                range_exprs.append(f"(salaryMax >= {min_val} and salaryMin <= {max_val})")
    return "(" + " or ".join(range_exprs) + ")" if range_exprs else None


def _build_week_days_filter_milvus_sxxz(week_work_days: Optional[int]) -> Optional[str]:
    if week_work_days in (1, 2, 3, 4):
        return f"weekWorkDays == {week_work_days}"
    return None


def _build_internship_months_filter_milvus_sxxz(internship_months: Optional[List[int]]) -> Optional[str]:
    if not internship_months or 0 in internship_months:
        return None
    valid = [m for m in internship_months if isinstance(m, int) and 0 <= m <= 5]
    if len(valid) == 1:
        return f"internshipMonth == {valid[0]}"
    elif len(valid) > 1:
        return f"internshipMonth in {valid}"
    return None


def _build_is_remote_filter_milvus_sxxz(is_remote: Optional[int]) -> Optional[str]:
    if is_remote == 1:
        return "is_remote == false"
    elif is_remote == 2:
        return "is_remote == true"
    return None


def _build_work_nature_filter_milvus_sxxz(work_nature: Optional[int]) -> Optional[str]:
    if work_nature in (2, 11):
        return f"workNature == {work_nature}"
    return "workNature == 2 or workNature == 11"


def build_education_filter_milvus(education: Optional[int]) -> Optional[str]:
    if education in (1, 2, 3, 4, 5):
        return f"education <= {education}"
    return None


def _build_refresh_time_filter_milvus(refresh_time: Optional[int]) -> Optional[str]:
    if refresh_time is None or not isinstance(refresh_time, int) or refresh_time == 0:
        return None
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    now = int(today_start.timestamp())
    mapping = {1: now - 86400, 2: now - 259200, 3: now - 604800, 4: now - 1209600, 5: now - 2592000}
    gte_ts = mapping.get(refresh_time)
    return f"updated_time >= {gte_ts}" if gte_ts else None


def _build_locations_filter_milvus(locations: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not locations or not isinstance(locations, list):
        return None
    location_exprs = []
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        city = _sanitize_str(loc.get("city"))
        county = _sanitize_str(loc.get("county"))
        if city and county and county != "全部":
            location_exprs.append(f'(city == "{city}" and county == "{county}")')
        elif county and county != "全部":
            location_exprs.append(f'county == "{county}"')
        elif city:
            location_exprs.append(f'city == "{city}"')
    return "(" + " or ".join(location_exprs) + ")" if location_exprs else None


def _build_gender_filter_milvus(gender: Optional[int]) -> Optional[str]:
    if gender in (1, 2):
        return f"gender == {gender}"
    return None


def _build_salary_settle_filter_milvus(salary_settle: Optional[str]) -> Optional[str]:
    s = _sanitize_str(salary_settle)
    if not s or s.upper() == "ANY":
        return None
    return f'salarySettle == "{s.upper()}"' if s.upper() in {"ORDER", "DAY", "WEEK", "MONTH", "OTHER"} else None


def _build_job_opening_state_filter_milvus(state: Optional[int]) -> Optional[str]:
    if state == 1:
        return "is_open == true"
    if state in (-1, 0):
        return "is_open == false"
    return None


def build_milvus_expr_sxxz(params: Dict[str, Any]) -> Optional[str]:
    return _and_expr((
        _build_work_nature_filter_milvus_sxxz(params.get("workNature")),
        _build_locations_filter_milvus(params.get("locations")),
        _build_salary_ranges_filter_milvus_sxxz(params.get("salaryRanges")),
        build_education_filter_milvus(params.get("education")),
        _build_week_days_filter_milvus_sxxz(params.get("weekWorkDays")),
        _build_internship_months_filter_milvus_sxxz(params.get("internshipMonths")),
        _build_is_remote_filter_milvus_sxxz(params.get("is_remote")),
        _build_refresh_time_filter_milvus(params.get("refreshTime")),
        _build_gender_filter_milvus(params.get("gender")),
        _build_salary_settle_filter_milvus(params.get("salarySettle")),
        _build_job_opening_state_filter_milvus(params.get("is_open")),
    ))


# ----------------------------- 简历处理（优化版） -----------------------------
async def build_resume_vector_async(resume: dict, params: Dict) -> np.ndarray:
    """异步构建简历向量"""
    vector_list = []
    weights = []

    basic = resume.get("basic_info", {})
    texts_to_embed = []
    weight_map = {}

    # 收集需要向量化的文本
    for field, weight in [("major", 1.0), ("courses", 1.0), ("intent", 3.0)]:
        text = basic.get(field)
        if text:
            if isinstance(text, list):
                text = ",".join(str(t) for t in text if t)
            if text.strip():
                texts_to_embed.append(text.strip())
                weight_map[text.strip()] = weight

    if not texts_to_embed:
        # 回退逻辑
        emb_map = get_local_embeddings_cache()
        other_vec = emb_map.get("其他")
        if other_vec and len(other_vec) == EMBEDDING_DIM:
            vec_np = np.asarray(other_vec, dtype=np.float32)
            norm = np.linalg.norm(vec_np)
            return (vec_np / norm).astype(np.float32) if norm > 0 else vec_np
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # 并发向量化
    embeddings = await AsyncEmbeddingClient.embed_texts(texts_to_embed)

    for text, emb in zip(texts_to_embed, embeddings):
        if emb and len(emb) == EMBEDDING_DIM:
            vector_list.append(np.asarray(emb, dtype=np.float32))
            weights.append(weight_map[text])

    if not vector_list:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    V = np.vstack(vector_list)
    W = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    sum_w = float(np.sum(W))
    if sum_w == 0:
        W = np.ones_like(W, dtype=np.float32)
        sum_w = float(np.sum(W))

    vec = np.sum(V * W, axis=0) / sum_w
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def build_resume_text(resume: dict) -> str:
    """构建简历稀疏文本（纯计算，无需异步）"""
    parts = []
    basic = resume.get("basic_info", {})

    def add(text, weight=1):
        if not text:
            return
        if isinstance(text, list):
            text = " ".join(text)
        for _ in range(weight):
            parts.append(text)

    add(basic.get("major"), 3)
    add(basic.get("courses"), 2)
    return "".join(parts)


# ----------------------------- Milvus查询（优化版） -----------------------------
async def hybrid_recall_with_rrf_async(
        collection: Collection,
        resume_vec: np.ndarray,
        query_text: str,
        filters: Optional[str] = None,
        top_k: int = 300,
        candidate_limit: int = 500,
        rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """异步混合召回"""
    async with _sem_milvus:
        loop = asyncio.get_running_loop()

        def _sync_search():
            search_requests = []

            # 向量召回
            vector_request = AnnSearchRequest(
                data=[resume_vec.tolist()],
                anns_field="job_vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 128}},
                limit=candidate_limit,
                expr=filters
            )
            search_requests.append(vector_request)

            # BM25召回
            try:
                bm25_request = AnnSearchRequest(
                    data=[query_text],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=candidate_limit,
                    expr=filters
                )
                search_requests.append(bm25_request)
            except Exception as e:
                logger.warning(f"BM25 request failed: {e}")

            # RRF融合
            rrf_ranker = RRFRanker(rrf_k)
            search_results = collection.hybrid_search(
                reqs=search_requests,
                rerank=rrf_ranker,
                limit=candidate_limit,
                output_fields=["job_id", "job_name", "job_vector", "job_describe", "province", "city"]
            )

            # 收集结果
            results_map = {}
            for hits in search_results:
                for hit in hits:
                    job_id = hit.entity.get("job_id")
                    if job_id is None:
                        continue
                    job_id = int(job_id)
                    job_vector = np.array(hit.entity.get("job_vector"))
                    sim = float(cosine_sim(resume_vec, job_vector)) if job_vector.size > 0 else 0.0

                    if sim > 0.5:
                        results_map[job_id] = {
                            "job_id": job_id,
                            "job_name": hit.entity.get("job_name", ""),
                            "job_sim": sim,
                            "rrf_score": hit.score,
                            "job_describe": hit.entity.get("job_describe", ""),
                            "province": hit.entity.get("province", ""),
                            "city": hit.entity.get("city", "")
                        }

            return sorted(results_map.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

        return await loop.run_in_executor(get_thread_pool(), _sync_search)


async def fetch_jobs_by_id_async(collection: Collection, job_ids: List[int]) -> List[Dict]:
    """异步批量查询岗位详情"""
    if not job_ids:
        return []

    async with _sem_milvus:
        loop = asyncio.get_running_loop()

        def _sync_query():
            expr = f"job_id in {job_ids}"
            return collection.query(
                expr=expr,
                output_fields=["job_id", "job_name", "job_describe", "province", "city"]
            )

        return await loop.run_in_executor(get_thread_pool(), _sync_query)


# ----------------------------- 重排提示词构建 -----------------------------
def build_query_text_sxxz(resume: dict = None, max_total_length: int = 800) -> str:
    """
    构建实习校招重排提示词 - 优化版

    Args:
        resume: 简历字典
        job_info: 意向岗位列表
        max_total_length: 最大总长度限制（gte-rerank-v2建议不超过512 tokens，约800汉字）
    """
    parts = []

    def add(title: str, text: str, max_length: int = 200):
        """添加字段，带长度截断"""
        if not text:
            return
        if isinstance(text, list):
            text = ";".join(str(t) for t in text if t)
        text = str(text).strip()
        if not text:
            return

        # 截断过长文本，保留关键信息
        if len(text) > max_length:
            # 尝试在句子边界截断
            truncated = text[:max_length]
            last_punct = max(truncated.rfind('。'), truncated.rfind('；'), truncated.rfind('，'))
            if last_punct > max_length * 0.7:  # 如果找到合适的标点，在那里截断
                truncated = truncated[:last_punct + 1]
            text = truncated + "..."

        parts.append(f"{title}：{text}")

    # ---------- 2. 简历信息 - 限制数量和长度 ----------
    if resume and isinstance(resume, dict):
        basic = resume.get("basic_info", {}) or {}

        # 专业（核心）
        add("专业【核心】", basic.get("major"), max_length=50)

        # 项目经历 - 只取前2个，每个限制长度
        projects = resume.get("project_exp", []) or []
        if projects:
            # 按时间倒序，取最近2个
            recent_projects = projects[:2]
            project_summaries = []
            for proj in recent_projects:
                if isinstance(proj, dict):
                    content = proj.get("content", "") or ""
                    # 提取前50字作为摘要，避免过长
                    summary = content[:80] if len(content) > 80 else content
                    if summary:
                        project_summaries.append(summary)

            if project_summaries:
                add("项目经历【核心】", "；".join(project_summaries), max_length=250)

        # 技能 - 限制数量
        skills = resume.get("skills", [])
        if skills and isinstance(skills, list):
            # 最多取5个技能
            skill_str = "、".join([str(s) for s in skills[:5] if s])
            add("技能【重要】", skill_str, max_length=100)
        elif isinstance(skills, str) and skills.strip():
            add("技能【重要】", skills, max_length=100)

        # 校内实践 - 只取1个最新的
        campus_exp = resume.get("campus_exp", []) or []
        if campus_exp and isinstance(campus_exp, list):
            first_exp = campus_exp[0]
            if isinstance(first_exp, dict):
                content = first_exp.get("content", "") or ""
                if content:
                    add("校内实践【参考】", content[:100], max_length=100)

        # 自我评价 - 限制长度
        self_eval = resume.get("self", [])
        if isinstance(self_eval, list) and self_eval:
            self_str = " ".join([str(s) for s in self_eval if s])
            add("自我评价【参考】", self_str, max_length=150)
        elif isinstance(self_eval, str) and self_eval.strip():
            add("自我评价【参考】", self_eval, max_length=150)

    # ---------- 3. 统一指令（简化版，减少token）----------
    parts.append("匹配度评估：优先核心项，其次重要项，参考项辅助。")

    # ---------- 4. 最终长度控制 ----------
    result = "\n".join(parts)

    if len(result) > max_total_length:
        logger.warning(f"Query text too long ({len(result)} chars), truncating to {max_total_length}")
        # 保留高优先级部分（前面的部分），截断后面的参考信息
        result = result[:max_total_length] + "..."

    # 记录实际长度用于监控
    logger.debug(f"Built query text with length: {len(result)}")

    return result


# ----------------------------- 核心服务（优化版） -----------------------------
async def search_sxxz_service_optimized(filters: Dict) -> Dict[str, Any]:
    """实习校招岗位推荐 - 完全优化版"""
    start_time = time.time()
    global _milvus_collection
    collection = _milvus_collection
    if collection is None:
        loop = asyncio.get_running_loop()
        collection = await loop.run_in_executor(None, new_milvus_collection)
        _milvus_collection = collection

    resume = filters.get("resume")
    # job_info = filters.get("jobInfo", [])
    page = max(int(filters.get("pageNum", 1)), 1)
    page_size = max(int(filters.get("pageSize", 20)), 1)

    # 提前计算截断数量
    result_from = (page - 1) * page_size
    rerank_cutoff = max(50, result_from + page_size * 2)

    # ========== 1. 并发执行：简历召回 + 求职意向召回 ==========
    # 简历召回
    if resume:
        async def resume_recall():
            vec = await build_resume_vector_async(resume, filters)
            text = build_resume_text(resume)
            print("text:", text)
            filter_expr = build_milvus_expr_sxxz(filters)
            print("filter_expr:", filter_expr)
            results = await hybrid_recall_with_rrf_async(
                collection=collection,
                resume_vec=vec,
                query_text=text,
                filters=filter_expr,
                top_k=rerank_cutoff,
                candidate_limit=rerank_cutoff
            )
            return [r["job_id"] for r in results]

        resume_task = resume_recall()
    else:
        resume_task = asyncio.sleep(0)

    # 并行执行
    results = await resume_task
    resume_ids = results if resume else []

    logger.info(f"实习校招召回完成 - 简历: {len(resume_ids)}个, 耗时: {time.time() - start_time:.2f}s")

    # ========== 2. RRF融合 + 提前截断 ==========
    # rank_map = {k: v for k, v in (("resume", resume_ids), ("intent", intent_ids)) if v}
    #
    # if not rank_map:
    #     return {"jobs": [], "returnedCount": 0, "pageNum": page, "pageSize": page_size}
    #
    # ranked_ids, _ = rrf_fusion(rank_map, k=60)
    # rerank_ids = ranked_ids[:rerank_cutoff]  # 提前截断，减少后续处理
    rerank_ids = resume_ids[:rerank_cutoff]
    # ========== 3. 查询详情 + 异步重排 ==========
    job_docs = await fetch_jobs_by_id_async(collection, rerank_ids)

    id_to_doc = {int(doc["job_id"]): doc for doc in job_docs}
    candidates = [
        {
            "job_id": jid,
            "job_name": doc["job_name"],
            "job_describe": doc["job_describe"],
            "city": doc["city"],
        }
        for jid in rerank_ids
        if (doc := id_to_doc.get(jid)) is not None
    ]

    # 异步重排
    query_text = build_query_text_sxxz(resume)
    documents = [f"{c['job_name']}\n{c['job_describe']}" for c in candidates]

    rerank_results = await AsyncRerankClient.rerank(query_text, documents, top_k=len(candidates))

    if rerank_results:
        reranked = []
        for r in rerank_results:
            idx = r["index"]
            if 0 <= idx < len(candidates):
                item = candidates[idx].copy()
                item["rerank_score"] = r["score"]
                reranked.append(item)
    else:
        reranked = [dict(item, rerank_score=0.0) for item in candidates]

    # ========== 4. 分页 ==========
    start = (page - 1) * page_size
    end = start + page_size
    page_ids = reranked[start:end]

    total_time = time.time() - start_time
    logger.info(f"实习校招推荐完成 - 总耗时: {total_time:.2f}s, 返回: {len(page_ids)}个结果")

    return {
        "jobs": page_ids,
        "returnedCount": len(page_ids),
        "pageNum": page,
        "pageSize": page_size,
    }


# ----------------------------- API端点 -----------------------------
@app.post("/api/v1/jobs/list_recommend_Resume")
async def search_sxxz_with_resume(req: SearchRequestSXXZ):
    logger.info(f"进入实习校招岗位推荐系统(优化版)...")
    filters = _req_to_filters(req)
    if "is_open" not in filters or filters.get("is_open") is None:
        filters["is_open"] = 1
    return await search_sxxz_service_optimized(filters)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8125, log_level="info")
