# -*- coding: utf-8 -*-
"""
简历投递匹配分数服务（重构版）

设计目标：
1. 保留与旧版一致的 4 个接口、参数和响应结构。
2. 与岗位推荐服务协同：复用 common 中的 ResumeVectorBuilder / RerankQueryBuilder / AsyncRerankClient。
3. 面向真实招聘场景：使用多维信号（语义、技能覆盖、经历、意向、城市、教育）而非纯关键词。
"""
import json
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import aiomysql
import aiohttp
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    RESUME_TYPE_CAMPUS,
    RESUME_TYPE_PART_TIME,
    COARSE_SCORE_WEIGHT,
    RERANK_SCORE_WEIGHT,
    SERVICE_HOST,
    JOB_DETAIL_API_BASE,
    JOB_DETAIL_API_TOKEN,
)
from common import (
    logger,
    create_http_session,
    create_db_pool,
    AsyncEmbeddingClient,
    AsyncRerankClient,
    ResumeVectorBuilder,
    RerankQueryBuilder,
    build_resume_text,
    cosine_sim,
)


# ----------------------------- 生命周期 -----------------------------
class ServiceState:
    def __init__(self):
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[aiomysql.Pool] = None
        self.embedding_client: Optional[AsyncEmbeddingClient] = None
        self.rerank_client: Optional[AsyncRerankClient] = None

    async def initialize(self):
        logger.info("=== 投递匹配服务(v2)启动，初始化资源 ===")
        self.http_session = create_http_session()
        self.db_pool = await create_db_pool()
        self.embedding_client = AsyncEmbeddingClient(self.http_session)
        self.rerank_client = AsyncRerankClient(self.http_session)
        logger.info("=== 投递匹配服务(v2)初始化完成 ===")

    async def cleanup(self):
        logger.info("=== 投递匹配服务(v2)关闭，清理资源 ===")
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        if self.db_pool:
            self.db_pool.close()
            await self.db_pool.wait_closed()
        logger.info("=== 投递匹配服务(v2)资源清理完成 ===")


_state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _state.initialize()
    yield
    await _state.cleanup()


app = FastAPI(lifespan=lifespan)


# ----------------------------- 请求/响应模型（保持兼容） -----------------------------
class CampusDeliverRequest(BaseModel):
    resume_id: int
    job_name: str
    job_describe: str
    job_cities: Optional[List[str]] = None


class PartTimeDeliverRequest(BaseModel):
    resume_id: int
    job_name: str
    job_describe: str
    job_cities: Optional[List[str]] = None


class CampusDeliverByJobIdRequest(BaseModel):
    resume_id: int
    job_id: int
    api_token: Optional[str] = None


class PartTimeDeliverByJobIdRequest(BaseModel):
    resume_id: int
    job_id: int
    api_token: Optional[str] = None


class DeliverResponse(BaseModel):
    score: float


# ----------------------------- 数据访问 -----------------------------
async def get_attachment_resume_by_id(
    db_pool: aiomysql.Pool,
    resume_id: int,
    resume_type: int,
) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            sql = """
                SELECT id, user_id, name, resume_type, work_city,
                       preview_address, post_address
                FROM pr_resume_parse
                WHERE id = %s AND resume_type = %s AND is_deleted = 0
                LIMIT 1
            """
            await cur.execute(sql, (resume_id, resume_type))
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "name": row["name"] or "",
                "resume_type": row["resume_type"],
                "work_city": row["work_city"] or "",
                "preview_address": row["preview_address"],
                "post_address": row["post_address"],
            }


async def fetch_job_detail(job_id: int, api_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    url = f"{JOB_DETAIL_API_BASE}/{job_id}"
    token = api_token if api_token else JOB_DETAIL_API_TOKEN
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with _state.http_session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning(f"获取岗位详情失败: job_id={job_id}, status={resp.status}, body={body[:200]}")
                return None

            data = await resp.json()
            if data.get("code") == 0 and data.get("success"):
                return data.get("data")

            logger.warning(
                f"岗位详情接口返回失败: job_id={job_id}, "
                f"code={data.get('code')}, msg={data.get('message') or data.get('msg')}"
            )
            return None
    except Exception as e:
        logger.error(f"获取岗位详情异常: job_id={job_id}, error={str(e)}")
        return None


def parse_job_detail(data: Dict[str, Any]) -> Dict[str, str]:
    job_name = data.get("jobName", "")
    describe = data.get("describe", "")
    requirement = data.get("requirement", "")
    city = data.get("city", "")

    welfare_names = [w.get("welfareName", "") for w in data.get("welfareList", []) if w.get("welfareName")]
    tag_names = [t.get("name", "") for t in data.get("tags", []) if t.get("name")]

    parts = [f"岗位：{job_name}", describe, requirement]
    if welfare_names:
        parts.append("福利标签：" + "、".join(welfare_names))
    if tag_names:
        parts.append("岗位标签：" + "、".join(tag_names))

    job_describe = "\n".join([p.strip() for p in parts if p and p.strip()])
    return {"job_name": job_name, "job_describe": job_describe, "city": city}


# ----------------------------- 核心评分逻辑 -----------------------------
def _normalize_city(city: str) -> str:
    if not city:
        return ""
    cleaned = re.sub(r"[\s省市区县自治区特别行政区]", "", str(city).strip())
    cleaned = cleaned.replace("中华人民共和国", "")
    return cleaned


def _city_score(work_city: str, job_cities: Optional[List[str]]) -> float:
    if not work_city or not job_cities:
        return 0.6

    worker = _normalize_city(work_city)
    if not worker:
        return 0.6

    normalized = [_normalize_city(c) for c in job_cities if c]
    normalized = [c for c in normalized if c]
    if not normalized:
        return 0.6

    if any(worker == jc for jc in normalized):
        return 1.0
    if any(worker in jc or jc in worker for jc in normalized):
        return 0.85
    return 0.35


def _experience_score(resume_data: Dict[str, Any], job_text: str) -> float:
    project_exp = resume_data.get("project_exp", []) or []
    work_exp = resume_data.get("work_exp", []) or []
    intern_exp = resume_data.get("internship_exp", []) or []

    exp_count = len(project_exp) + len(work_exp) + len(intern_exp)
    base = 0.45 if exp_count == 0 else min(0.9, 0.55 + 0.08 * exp_count)

    # 场景校正：岗位描述强调资深/管理，而简历经历较少时适度降分
    senior_terms = ["资深", "高级", "负责人", "管理", "带团队", "5年", "3年"]
    if any(t in job_text for t in senior_terms) and exp_count <= 1:
        base -= 0.2

    return float(np.clip(base, 0.1, 1.0))


async def _semantic_signals(resume_data: Dict[str, Any], job_name: str, job_describe: str) -> Tuple[float, float, float]:
    """
    返回: (resume_vs_job, intent_vs_job, major_vs_job) ∈ [0,1]
    """
    job_text = f"岗位:{job_name}\n{job_describe[:800]}"

    resume_vector = await ResumeVectorBuilder.build_resume_vector(resume_data, _state.embedding_client)
    resume_sparse = build_resume_text(resume_data)

    basic = resume_data.get("basic_info", {}) or {}
    major = str(basic.get("major", "")).strip()
    intent = basic.get("intent", []) or basic.get("intent_positions", [])
    if isinstance(intent, list):
        intent_text = "、".join([str(x) for x in intent if x][:4])
    else:
        intent_text = str(intent)

    texts = [job_text]
    if resume_sparse:
        texts.append(resume_sparse)
    if major:
        texts.append(major)
    if intent_text:
        texts.append(intent_text)

    embeddings = await _state.embedding_client.embed_texts(texts)
    job_vec = np.asarray(embeddings[0], dtype=np.float32) if embeddings and embeddings[0] else None

    resume_vs_job = 0.0
    intent_vs_job = 0.0
    major_vs_job = 0.0

    if job_vec is not None and np.linalg.norm(job_vec) > 0:
        idx = 1
        if resume_sparse and len(embeddings) > idx and embeddings[idx]:
            resume_text_vec = np.asarray(embeddings[idx], dtype=np.float32)
            text_sim = float(cosine_sim(resume_text_vec, job_vec))
            vec_sim = float(cosine_sim(resume_vector, job_vec)) if np.linalg.norm(resume_vector) > 0 else 0.0
            resume_vs_job = max(0.0, min(1.0, 0.6 * vec_sim + 0.4 * text_sim))
            idx += 1
        else:
            vec_sim = float(cosine_sim(resume_vector, job_vec)) if np.linalg.norm(resume_vector) > 0 else 0.0
            resume_vs_job = max(0.0, min(1.0, vec_sim))

        if major and len(embeddings) > idx and embeddings[idx]:
            major_vs_job = max(0.0, min(1.0, float(cosine_sim(np.asarray(embeddings[idx], dtype=np.float32), job_vec))))
            idx += 1

        if intent_text and len(embeddings) > idx and embeddings[idx]:
            intent_vs_job = max(0.0, min(1.0, float(cosine_sim(np.asarray(embeddings[idx], dtype=np.float32), job_vec))))

    return resume_vs_job, intent_vs_job, major_vs_job


async def _rerank_score(resume_data: Dict[str, Any], job_name: str, job_describe: str) -> float:
    """
    使用与推荐服务一致的 Query 构造 + rerank 模型。
    为避免单候选打分偏低，加入锚点文档做标尺。
    """
    query = RerankQueryBuilder.build_query(
        resume=resume_data,
        job_titles=[job_name],
        job_info=None,
    )
    if not query:
        query = build_resume_text(resume_data)[:400]

    target_doc = f"岗位名称：{job_name}\n岗位描述：{job_describe[:1200]}"
    anchor_docs = [
        "岗位名称：基础行政助理\n岗位描述：负责日常表格整理、基础沟通协调，技能要求较低。",
        "岗位名称：跨领域高级研发负责人\n岗位描述：要求多年大型系统架构经验、团队管理和复杂项目交付。",
    ]
    documents = [target_doc] + anchor_docs

    results = await _state.rerank_client.rerank(query=query, documents=documents, top_k=len(documents))
    if not results:
        return 0.0

    hit = next((r for r in results if int(r.get("index", -1)) == 0), None)
    if not hit:
        return 0.0

    return float(hit.get("normalized_score", hit.get("score", 0.0)))


async def calc_delivery_score(
    resume_data: Dict[str, Any],
    work_city: str,
    job_name: str,
    job_describe: str,
    job_cities: Optional[List[str]] = None,
) -> float:
    """计算单岗位投递匹配分数（0~1）。"""
    if not resume_data or not job_name:
        return 0.0

    resume_vs_job, intent_vs_job, major_vs_job = await _semantic_signals(resume_data, job_name, job_describe)
    city = _city_score(work_city, job_cities)
    exp = _experience_score(resume_data, f"{job_name}\n{job_describe}")

    # 粗排：更可解释，覆盖真实招聘中的关键维度
    coarse_score = (
        0.40 * resume_vs_job +
        0.20 * intent_vs_job +
        0.15 * major_vs_job +
        0.15 * city +
        0.10 * exp
    )

    rerank_score = await _rerank_score(resume_data, job_name, job_describe)

    # 与推荐系统一致的融合策略，并加协同校正避免“推荐高、投递低”
    final_score = COARSE_SCORE_WEIGHT * coarse_score + RERANK_SCORE_WEIGHT * rerank_score

    # 协同校正：语义很高时给出保底，避免投递分严重背离推荐排序
    if resume_vs_job >= 0.72 and intent_vs_job >= 0.60:
        floor = min(0.92, coarse_score * 0.92)
        final_score = max(final_score, floor)

    return round(float(np.clip(final_score, 0.0, 1.0)), 4)


async def _get_resume_data(resume_id: int, resume_type: int) -> Optional[Dict[str, Any]]:
    row = await get_attachment_resume_by_id(_state.db_pool, resume_id, resume_type)
    if not row or not row.get("post_address"):
        return None

    post_address = row.get("post_address")
    try:
        resume_data = json.loads(post_address) if isinstance(post_address, str) else post_address
    except json.JSONDecodeError:
        logger.warning(f"简历 JSON 解析失败: resume_id={resume_id}")
        return None

    if not isinstance(resume_data, dict):
        return None

    return {
        "resume_data": resume_data,
        "work_city": row.get("work_city", ""),
    }


# ----------------------------- API 端点（保持 4 个接口） -----------------------------
@app.post("/api/v1/jobs/delivery_score_campus", response_model=DeliverResponse)
async def delivery_score_campus(req: CampusDeliverRequest) -> DeliverResponse:
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_CAMPUS)
    if not info:
        return DeliverResponse(score=0.0)

    score = await calc_delivery_score(
        resume_data=info["resume_data"],
        work_city=info["work_city"],
        job_name=req.job_name,
        job_describe=req.job_describe,
        job_cities=req.job_cities,
    )
    logger.info(f"投递分数(校招): resume_id={req.resume_id}, job={req.job_name}, score={score}")
    return DeliverResponse(score=score)


@app.post("/api/v1/jobs/delivery_score_part_time", response_model=DeliverResponse)
async def delivery_score_part_time(req: PartTimeDeliverRequest) -> DeliverResponse:
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_PART_TIME)
    if not info:
        return DeliverResponse(score=0.0)

    score = await calc_delivery_score(
        resume_data=info["resume_data"],
        work_city=info["work_city"],
        job_name=req.job_name,
        job_describe=req.job_describe,
        job_cities=req.job_cities,
    )
    logger.info(f"投递分数(兼职): resume_id={req.resume_id}, job={req.job_name}, score={score}")
    return DeliverResponse(score=score)


@app.post("/api/v1/jobs/delivery_score_campus_by_job_id", response_model=DeliverResponse)
async def delivery_score_campus_by_job_id(req: CampusDeliverByJobIdRequest) -> DeliverResponse:
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_CAMPUS)
    if not info:
        return DeliverResponse(score=0.0)

    job_data = await fetch_job_detail(req.job_id, req.api_token)
    if not job_data:
        return DeliverResponse(score=0.0)

    parsed = parse_job_detail(job_data)
    job_cities = [parsed["city"]] if parsed.get("city") else None

    score = await calc_delivery_score(
        resume_data=info["resume_data"],
        work_city=info["work_city"],
        job_name=parsed["job_name"],
        job_describe=parsed["job_describe"],
        job_cities=job_cities,
    )
    logger.info(f"投递分数(校招/by_job_id): resume_id={req.resume_id}, job_id={req.job_id}, score={score}")
    return DeliverResponse(score=score)


@app.post("/api/v1/jobs/delivery_score_part_time_by_job_id", response_model=DeliverResponse)
async def delivery_score_part_time_by_job_id(req: PartTimeDeliverByJobIdRequest) -> DeliverResponse:
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_PART_TIME)
    if not info:
        return DeliverResponse(score=0.0)

    job_data = await fetch_job_detail(req.job_id, req.api_token)
    if not job_data:
        return DeliverResponse(score=0.0)

    parsed = parse_job_detail(job_data)
    job_cities = [parsed["city"]] if parsed.get("city") else None

    score = await calc_delivery_score(
        resume_data=info["resume_data"],
        work_city=info["work_city"],
        job_name=parsed["job_name"],
        job_describe=parsed["job_describe"],
        job_cities=job_cities,
    )
    logger.info(f"投递分数(兼职/by_job_id): resume_id={req.resume_id}, job_id={req.job_id}, score={score}")
    return DeliverResponse(score=score)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=8234, log_level="info")
