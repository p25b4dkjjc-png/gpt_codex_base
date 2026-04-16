# -*- coding: utf-8 -*-
"""
兼职岗位推荐服务 - 优化版
优化点：改进多通道（简历/求职意向）融合策略，解决单通道主导问题
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
import aiomysql

from config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME,
    PART_TIME_SERVICE_PORT, SERVICE_HOST,
    WORK_TYPE_PART_TIME, WORK_NATURE_PART_TIME,
    RESUME_TYPE_PART_TIME, EMBEDDING_CONCURRENCY, MILVUS_CONCURRENCY,
    THRESHOLD, RRF_K_DEFAULT, RERANK_CUTOFF_MULTIPLIER, MIN_SIMILARITY_THRESHOLD
)
from common import (
    logger, LocationItem, _req_to_filters,
    create_http_session, create_db_pool, load_local_embeddings_cache,
    AsyncEmbeddingClient, AsyncRerankClient,
    ResumeVectorBuilder, OnlineResumeVectorBuilder,
    build_resume_text,
    IntelligentReranker, OnlineIntelligentReranker,
    hybrid_recall_with_rrf_async, fetch_jobs_by_id_async,
    get_resume_list_by_user_attachment, get_resume_list_by_user_online,
    _build_locations_filter_milvus, _build_refresh_time_filter_milvus,
    _build_gender_filter_milvus, _build_salary_settle_filter_milvus,
    _build_job_opening_state_filter_milvus, build_education_filter_milvus,
    _and_expr, _dedupe_keep_order, cosine_sim
)


# ----------------------------- 服务状态管理 -----------------------------
class ServiceState:
    """服务状态管理类 - 每个服务独立管理自己的资源"""

    def __init__(self):
        self.http_session = None
        self.db_pool = None
        self.milvus_collection = None
        self.embedding_client = None
        self.rerank_client = None
        self.intelligent_reranker = None
        self.online_intelligent_reranker = None
        self.local_emb_cache = None
        self._sem_milvus = None

    async def initialize(self):
        """初始化所有资源"""
        logger.info("=== 兼职服务启动，初始化资源 ===")

        # 加载本地缓存
        self.local_emb_cache = load_local_embeddings_cache()

        # 创建 HTTP Session
        self.http_session = create_http_session()

        # 创建数据库连接池
        self.db_pool = await create_db_pool()

        # 连接 Milvus
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.milvus_collection = Collection(COLLECTION_NAME)
        try:
            self.milvus_collection.load()
        except Exception:
            pass

        # 创建客户端实例
        self.embedding_client = AsyncEmbeddingClient(self.http_session, concurrency=EMBEDDING_CONCURRENCY)
        self.rerank_client = AsyncRerankClient(self.http_session)
        self.intelligent_reranker = IntelligentReranker(self.rerank_client)
        self.online_intelligent_reranker = OnlineIntelligentReranker(self.rerank_client)

        # 并发控制
        self._sem_milvus = asyncio.Semaphore(MILVUS_CONCURRENCY)

        logger.info("=== 兼职服务初始化完成 ===")

    async def cleanup(self):
        """清理资源"""
        logger.info("=== 兼职服务关闭，清理资源 ===")
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        if self.db_pool:
            self.db_pool.close()
            await self.db_pool.wait_closed()
        logger.info("=== 兼职服务资源清理完成 ===")


# 全局状态实例
_state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    await _state.initialize()
    yield
    await _state.cleanup()


# ----------------------------- FastAPI 应用 -----------------------------
app = FastAPI(lifespan=lifespan)


# ----------------------------- Pydantic 模型 -----------------------------
class SearchPartRequest(BaseModel):
    user_id: int
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


# ----------------------------- 过滤条件构建（兼职专用） -----------------------------
def _build_week_days_filter_milvus(week_work_days: Optional[int]) -> Optional[str]:
    """构建每周工作天数过滤条件（兼职）"""
    if week_work_days is None or not isinstance(week_work_days, int) or week_work_days == 0:
        return None
    if week_work_days == 1:
        return "weekWorkDays in [1, 2]"
    elif week_work_days == 3:
        return "weekWorkDays in [2, 3]"
    elif week_work_days == 4:
        return "weekWorkDays == 4"
    return None


def _build_work_nature_filter_milvus(work_nature: Optional[int]) -> Optional[str]:
    """构建工作性质过滤条件（兼职）"""
    return f"workNature == {WORK_NATURE_PART_TIME}"


def build_milvus_expr(params: Dict[str, Any]) -> Optional[str]:
    """构建兼职 Milvus 过滤表达式"""
    return _and_expr((
        _build_work_nature_filter_milvus(params.get("workNature")),
        _build_locations_filter_milvus(params.get("locations")),
        _build_week_days_filter_milvus(params.get("weekWorkDays")),
        _build_refresh_time_filter_milvus(params.get("refreshTime")),
        _build_gender_filter_milvus(params.get("gender")),
        _build_salary_settle_filter_milvus(params.get("salarySettle")),
        _build_job_opening_state_filter_milvus(params.get("is_open")),
        build_education_filter_milvus(params.get("education")),
    ))


# ----------------------------- Milvus 查询（兼职专用） -----------------------------
async def milvus_hybrid_search_parttime_async(
        collection,
        query_texts: List[str],
        embedding_client: AsyncEmbeddingClient,
        filters: Dict[str, Any],
        page: int = 1,
        size: int = 20
) -> List[int]:
    """兼职异步混合搜索"""
    if not query_texts:
        return []

    result_from = (page - 1) * size
    target_position = page * size
    candidate_limit = min(600, target_position + 300)

    embeddings = await embedding_client.embed_texts(query_texts)
    valid_embeddings = [emb for emb in embeddings if emb and len(emb) > 0]

    if not valid_embeddings:
        return []

    filter_expr = build_milvus_expr(filters)

    search_requests = []
    for vec in valid_embeddings:
        vector_request = AnnSearchRequest(
            data=[vec],
            anns_field="job_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 128}},
            limit=candidate_limit,
            expr=filter_expr
        )
        search_requests.append(vector_request)

    for text in query_texts:
        try:
            bm25_request = AnnSearchRequest(
                data=[text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=candidate_limit,
                expr=filter_expr
            )
            search_requests.append(bm25_request)
        except Exception as e:
            logger.warning(f"BM25 request failed: {e}")

    async with _state._sem_milvus:
        loop = asyncio.get_running_loop()

        def _sync_search():
            rrf_ranker = RRFRanker(RRF_K_DEFAULT)
            search_results = collection.hybrid_search(
                reqs=search_requests,
                rerank=rrf_ranker,
                limit=candidate_limit,
                output_fields=["job_id", "job_vector"]
            )

            all_ids = []
            user_vectors = valid_embeddings

            for hits in search_results:
                for hit in hits:
                    job_id = hit.entity.get("job_id")
                    job_vector = np.array(hit.entity.get("job_vector"))

                    if job_id:
                        if job_vector.size > 0:
                            max_sim = max(cosine_sim(uv, job_vector) for uv in user_vectors)
                            if max_sim > MIN_SIMILARITY_THRESHOLD:
                                all_ids.append(int(job_id))
                        else:
                            all_ids.append(int(job_id))

            return all_ids

        all_ids = await loop.run_in_executor(None, _sync_search)

    unique_ids = _dedupe_keep_order(all_ids)
    cutoff = min(len(unique_ids), max(100, result_from + size * RERANK_CUTOFF_MULTIPLIER))
    return unique_ids[:cutoff]


# ----------------------------- 结果融合优化工具函数 -----------------------------

def min_max_normalize(score_map: Dict[int, Tuple[float, Dict]]) -> Dict[int, Tuple[float, Dict]]:
    """
    Min-Max 归一化 - 解决不同重排器打分尺度不一致问题
    将分数映射到 [0, 1] 区间，消除跨通道分数偏差
    """
    if not score_map:
        return {}

    scores = [s for s, _ in score_map.values()]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        # 所有分数相同，统一设为 0.5
        return {k: (0.5, v) for k, (s, v) in score_map.items()}

    normalized = {}
    for job_id, (score, info) in score_map.items():
        norm_score = (score - min_s) / (max_s - min_s)
        normalized[job_id] = (round(norm_score, 4), info)

    return normalized


def balanced_fusion_with_quota(
        resume_norm: Dict[int, Tuple[float, Dict]],
        intent_norm: Dict[int, Tuple[float, Dict]],
        final_limit: int = 100,
        min_quota: float = 0.25,  # 每通道保底 25%
        dual_channel_bonus: float = 0.15
) -> List[Tuple[int, float, str, Dict, float, float]]:
    """
    配额保底 + 双通道激励融合策略

    策略说明：
    1. 分通道归一化后，各自保底 min_quota 比例的结果
    2. 双通道命中的岗位给予额外 bonus
    3. 剩余 slots 按融合分数公平竞争
    4. 最终按融合分数排序

    Returns:
        List of (job_id, final_score, source, info, resume_score, intent_score)
    """
    if not resume_norm and not intent_norm:
        return []

    # 计算各通道保底数量
    min_count = int(final_limit * min_quota)

    # 分类：仅简历、仅意向、双通道
    resume_only_ids = set(resume_norm.keys()) - set(intent_norm.keys())
    intent_only_ids = set(intent_norm.keys()) - set(resume_norm.keys())
    both_ids = set(resume_norm.keys()) & set(intent_norm.keys())

    logger.info(f"融合前统计 - 仅简历:{len(resume_only_ids)}, 仅意向:{len(intent_only_ids)}, 双通道:{len(both_ids)}")

    # 准备各类型结果
    resume_only_items = [
        (jid, resume_norm[jid][0], "resume", resume_norm[jid][1], resume_norm[jid][0], 0.0)
        for jid in resume_only_ids
    ]
    intent_only_items = [
        (jid, intent_norm[jid][0], "intent", intent_norm[jid][1], 0.0, intent_norm[jid][0])
        for jid in intent_only_ids
    ]
    both_items = []
    for jid in both_ids:
        r_score, r_info = resume_norm[jid]
        i_score, i_info = intent_norm[jid]

        # 双通道融合：加权平均 + 双通道bonus
        # 策略：取最高分作为基础，副通道给予一定权重加成
        if r_score >= i_score:
            primary, secondary = r_score, i_score
            primary_source = "resume"
        else:
            primary, secondary = i_score, r_score
            primary_source = "intent"

        # 融合公式：主通道 + 副通道 * 0.3，然后给双通道bonus
        fused = primary + secondary * THRESHOLD.SECONDARY_WEIGHT
        final_score = min(1.0, fused * (1 + dual_channel_bonus))

        # 优先使用主通道的 info，但标记为双通道
        info_copy = dict(r_info if primary_source == "resume" else i_info)
        info_copy["dual_channel"] = True
        info_copy["resume_score"] = r_score
        info_copy["intent_score"] = i_score

        both_items.append((jid, final_score, "both", info_copy, r_score, i_score))

    # 排序
    resume_only_items.sort(key=lambda x: x[1], reverse=True)
    intent_only_items.sort(key=lambda x: x[1], reverse=True)
    both_items.sort(key=lambda x: x[1], reverse=True)

    # 第一阶段：保底配额
    result = []
    result.extend(resume_only_items[:min_count])
    result.extend(intent_only_items[:min_count])

    # 双通道结果优先进入（不受保底限制，但去重）
    existing_ids = {x[0] for x in result}
    for item in both_items:
        if item[0] not in existing_ids and len(result) < final_limit:
            result.append(item)
            existing_ids.add(item[0])

    # 第二阶段：填充剩余 slots
    remaining = final_limit - len(result)
    if remaining > 0:
        # 合并剩余候选
        remaining_candidates = []
        # 简历剩余
        for item in resume_only_items[min_count:]:
            if item[0] not in existing_ids:
                remaining_candidates.append(item)
        # 意向剩余
        for item in intent_only_items[min_count:]:
            if item[0] not in existing_ids:
                remaining_candidates.append(item)

        # 按分数排序，取前 remaining 个
        remaining_candidates.sort(key=lambda x: x[1], reverse=True)
        result.extend(remaining_candidates[:remaining])

    # 最终排序
    result.sort(key=lambda x: x[1], reverse=True)

    # 统计日志
    from_resume = sum(1 for x in result if x[2] == "resume")
    from_intent = sum(1 for x in result if x[2] == "intent")
    from_both = sum(1 for x in result if x[2] == "both")
    logger.info(f"融合后分布 - 仅简历:{from_resume}, 仅意向:{from_intent}, 双通道:{from_both}, 总计:{len(result)}")

    return result


# ----------------------------- 核心业务逻辑 -----------------------------
async def search_parttime_service_optimized(filters: Dict) -> Dict[str, Any]:
    """兼职岗位推荐 - 多简历独立召回重排后融合（优化版）"""
    start_time = time.time()
    collection = _state.milvus_collection

    # 获取用户简历信息（兼职类型 resume_type=2）
    user_id = filters.get("user_id")
    resume_list = await get_resume_list_by_user_attachment(_state.db_pool, user_id=user_id,
                                                           resume_type=RESUME_TYPE_PART_TIME)
    logger.info(f"简历查询完成: 共{len(resume_list)}条记录")

    job_titles = filters.get("jobTitles", [])
    page = max(int(filters.get("pageNum", 1)), 1)
    page_size = max(int(filters.get("pageSize", 20)), 1)

    # 提前计算截断数量
    result_from = (page - 1) * page_size
    rerank_cutoff = max(50, result_from + page_size * RERANK_CUTOFF_MULTIPLIER)

    # ========== 1. 简历部分：每份简历独立召回+重排 ==========
    async def process_single_resume(resume_item: Dict) -> List[Dict]:
        """处理单份附件简历"""
        post_address = resume_item.get("post_address")
        if not post_address:
            return []

        try:
            if isinstance(post_address, str):
                resume_data = json.loads(post_address)
            else:
                resume_data = post_address
        except json.JSONDecodeError:
            logger.warning(f"简历解析失败: resume_id={resume_item.get('id')}")
            return []

        # 为该简历构建特定的过滤条件
        resume_filters = {
            'is_open': filters.get("is_open"),
            'workNature': filters.get("workNature")
        }

        work_city_str = resume_item.get("work_city")
        if work_city_str and isinstance(work_city_str, str):
            cities = [city.strip() for city in work_city_str.split(",") if city.strip()]
            if cities:
                resume_filters["locations"] = [{"city": city} for city in cities]

        # 召回
        vec = await ResumeVectorBuilder.build_resume_vector(resume_data, _state.embedding_client, filters)
        text = build_resume_text(resume_data)
        filter_expr = build_milvus_expr(resume_filters)
        logger.info(f"简历{resume_item.get('id')}过滤filter_expr: {filter_expr}")

        recall_results = await hybrid_recall_with_rrf_async(
            collection=collection,
            resume_vec=vec,
            query_text=text,
            embedding_client=_state.embedding_client,
            filters=filter_expr,
            top_k=rerank_cutoff,
            candidate_limit=rerank_cutoff
        )

        if not recall_results:
            return []

        recall_ids = [r["job_id"] for r in recall_results]
        job_docs = await fetch_jobs_by_id_async(collection, recall_ids)
        id_to_doc = {int(doc["job_id"]): doc for doc in job_docs}

        candidates = [
            {
                "job_id": jid,
                "job_name": doc["job_name"],
                "job_describe": doc["job_describe"],
                "city": doc["city"],
            }
            for jid in recall_ids
            if (doc := id_to_doc.get(jid)) is not None
        ]

        # 独立重排
        reranked = await _state.intelligent_reranker.rerank(
            resume=resume_data,
            candidates=candidates,
            job_titles=None,
            job_info=None,
            resume_weight=1.0,
            intent_weight=0.0,
            top_k=rerank_cutoff
        )

        return reranked

    async def process_online_resume(resume_item: Dict) -> List[Dict]:
        """处理单份在线简历"""
        # 召回
        vec = await OnlineResumeVectorBuilder.build_vector(resume_item, _state.embedding_client)
        text = OnlineResumeVectorBuilder.build_sparse_text(resume_item)

        # 构造过滤条件
        resume_filters = {
            'is_open': filters.get("is_open"),
            'workNature': filters.get("workNature")
        }
        work_city_str = resume_item.get("work_city")
        if work_city_str and isinstance(work_city_str, str):
            cities = [city.strip() for city in work_city_str.split(",") if city.strip()]
            if cities:
                resume_filters["locations"] = [{"city": city} for city in cities]

        filter_expr = build_milvus_expr(resume_filters)
        logger.info(f"在线简历{resume_item.get('id')}过滤filter_expr: {filter_expr}")

        # 混合召回
        recall_results = await hybrid_recall_with_rrf_async(
            collection=collection,
            resume_vec=vec,
            query_text=text,
            embedding_client=_state.embedding_client,
            filters=filter_expr,
            top_k=rerank_cutoff,
            candidate_limit=rerank_cutoff
        )

        if not recall_results:
            return []

        recall_ids = [r["job_id"] for r in recall_results]
        job_docs = await fetch_jobs_by_id_async(collection, recall_ids)
        id_to_doc = {int(doc["job_id"]): doc for doc in job_docs}

        candidates = [
            {
                "job_id": jid,
                "job_name": doc["job_name"],
                "job_describe": doc["job_describe"],
                "city": doc["city"],
            }
            for jid in recall_ids
            if (doc := id_to_doc.get(jid)) is not None
        ]

        # 重排
        reranked = await _state.online_intelligent_reranker.rerank(
            resume=resume_item,
            candidates=candidates,
            job_titles=None,
            job_info=None,
            resume_weight=1.0,
            intent_weight=0.0,
            top_k=rerank_cutoff
        )

        return reranked

    # 并发处理所有简历
    online_resume_list = await get_resume_list_by_user_online(_state.db_pool, user_id=user_id,
                                                              work_type=WORK_TYPE_PART_TIME)
    logger.info(f"在线简历查询完成: 共{len(online_resume_list)}条记录")

    resume_tasks = [process_single_resume(item) for item in resume_list]
    resume_tasks += [process_online_resume(item) for item in online_resume_list]
    resume_results = await asyncio.gather(*resume_tasks, return_exceptions=True)

    # 收集所有简历结果，同岗位取最高final_score
    resume_best_scores = {}

    for idx, result in enumerate(resume_results):
        if isinstance(result, Exception):
            logger.error(f"简历{idx}处理失败: {result}")
            continue
        if result:
            for r in result:
                job_id = r["job_id"]
                final_score = r.get("final_score", 0)
                if job_id not in resume_best_scores or final_score > resume_best_scores[job_id][0]:
                    resume_best_scores[job_id] = (final_score, r)

    resume_list_scored = [
        (job_id, score, info) for job_id, (score, info) in resume_best_scores.items()
    ]

    logger.info(f"简历独立重排完成: {len(resume_list_scored)}个唯一岗位")

    # ========== 2. 意向部分：独立召回+重排 ==========
    intent_list_scored = []

    if job_titles:
        intent_ids = await milvus_hybrid_search_parttime_async(
            collection=collection,
            query_texts=job_titles if isinstance(job_titles, list) else [job_titles],
            embedding_client=_state.embedding_client,
            filters=filters,
            page=1,
            size=rerank_cutoff
        )

        if intent_ids:
            job_docs = await fetch_jobs_by_id_async(collection, intent_ids)
            id_to_doc = {int(doc["job_id"]): doc for doc in job_docs}

            candidates = [
                {
                    "job_id": jid,
                    "job_name": doc["job_name"],
                    "job_describe": doc["job_describe"],
                    "city": doc["city"],
                }
                for jid in intent_ids
                if (doc := id_to_doc.get(jid)) is not None
            ]

            # 意向独立重排
            reranked = await _state.intelligent_reranker.rerank(
                resume=None,
                candidates=candidates,
                job_titles=job_titles if isinstance(job_titles, list) else [job_titles] if job_titles else None,
                job_info=None,
                resume_weight=0.0,
                intent_weight=1.0,
                top_k=rerank_cutoff
            )

            for r in reranked:
                intent_list_scored.append((
                    r["job_id"],
                    r.get("final_score", 0),
                    r
                ))

            logger.info(f"意向重排完成: {len(intent_list_scored)}个结果")

    # ========== 3. 融合（优化版：配额保底 + 归一化） ==========
    resume_score_map = {job_id: (score, info) for job_id, score, info in resume_list_scored}
    intent_score_map = {job_id: (score, info) for job_id, score, info in intent_list_scored}

    logger.info(f"重排完成 - 简历通道:{len(resume_score_map)}个, 意向通道:{len(intent_score_map)}个")

    # 1. 跨通道分数归一化（解决打分尺度不一致问题）
    resume_norm = min_max_normalize(resume_score_map)
    intent_norm = min_max_normalize(intent_score_map)

    # 2. 配额保底 + 双通道激励融合
    final_ranked = balanced_fusion_with_quota(
        resume_norm=resume_norm,
        intent_norm=intent_norm,
        final_limit=rerank_cutoff,
        min_quota=0.30,  # 每通道保底 30%
        dual_channel_bonus=THRESHOLD.DUAL_CHANNEL_BONUS
    )

    # 3. 阈值过滤（在归一化后应用，更公平）
    filtered_results = []
    for item in final_ranked:
        job_id, final_score, source, info, resume_score, intent_score = item

        # 根据来源应用不同阈值
        if source == "both":
            threshold = THRESHOLD.FUSION_MIN_SCORE.get("dual_channel", 0.35)
        elif source == "resume":
            threshold = THRESHOLD.FUSION_MIN_SCORE.get("resume_only", 0.20)
        else:  # intent
            threshold = THRESHOLD.FUSION_MIN_SCORE.get("intent_only", 0.20)

        if final_score >= threshold:
            filtered_results.append(item)

    # 保底机制：如果过滤后结果过少，放宽阈值
    if len(filtered_results) < 10 and len(final_ranked) > 10:
        logger.warning(f"阈值过滤后结果过少({len(filtered_results)})，启用保底机制")
        filtered_results = final_ranked[:max(20, page_size * 2)]

    final_job_ids = [item[0] for item in filtered_results[:rerank_cutoff]]

    logger.info(
        f"融合完成 - 原始简历:{len(resume_score_map)}, 原始意向:{len(intent_score_map)}, "
        f"归一化后融合:{len(final_ranked)}, 阈值过滤后:{len(filtered_results)}, "
        f"最终截断:{len(final_job_ids)}"
    )

    from_resume = sum(1 for x in final_ranked if x[2] == "resume")
    from_intent = sum(1 for x in final_ranked if x[2] == "intent")
    from_both = sum(1 for x in final_ranked if x[2] == "both")
    logger.info(
        f"来源统计 - 仅简历:{from_resume}, "
        f"仅意向:{from_intent}, "
        f"双通道:{from_both}"
    )

    # ========== 4. 查询最终详情并返回 ==========
    if not final_job_ids:
        return {"ids": [], "returnedCount": 0, "pageNum": page, "pageSize": page_size}

    job_docs = await fetch_jobs_by_id_async(collection, final_job_ids)
    id_to_doc = {int(doc["job_id"]): doc for doc in job_docs}

    final_jobs = []
    final_ids = []
    for item in filtered_results[:rerank_cutoff]:
        job_id, final_score, source, info, resume_score, intent_score = item
        doc = id_to_doc.get(job_id)
        if not doc:
            continue
        final_ids.append(job_id)
        # final_jobs.append({
        #     "job_id": job_id,
        #     "job_name": doc.get("job_name"),
        #     "job_describe": doc.get("job_describe"),
        #     "city": doc.get("city"),
        #     "province": doc.get("province", ""),
        #     "final_score": round(final_score, 4),
        #     "resume_score": round(float(resume_score), 4),
        #     "intent_score": round(float(intent_score), 4),
        #     "source": source,
        #     "dual_channel": source == "both"
        # })

    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    # page_results = final_jobs[start:end]
    page_results = final_ids[start:end]
    total_time = time.time() - start_time
    logger.info(f"兼职推荐完成 - 总耗时: {total_time:.2f}s, 返回: {len(page_results)}个结果")

    return {
        "ids": page_results,
        "returnedCount": len(page_results),
        "pageNum": page,
        "pageSize": page_size,
    }


# ----------------------------- API 端点 -----------------------------
@app.post("/api/v1/part-time-jobs/search_with_Resume")
async def search_parttime_with_resume(req: SearchPartRequest):
    """兼职岗位推荐接口"""
    filters = _req_to_filters(req)
    if "is_open" not in filters or filters.get("is_open") is None:
        filters["is_open"] = 1
    filters["workNature"] = WORK_NATURE_PART_TIME
    return await search_parttime_service_optimized(filters)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=PART_TIME_SERVICE_PORT, log_level="info")
