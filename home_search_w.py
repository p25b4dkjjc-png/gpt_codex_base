"""
小店智能推荐服务 API - 首页统一搜索版（生产级 v2.6）
核心修复：类别词包含匹配 + 关键词同义词扩展 + 强探索型硬过滤
"""
import csv
import hashlib
import json
import logging
import math
import os
import re
import time
import threading
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple, Set

import dashscope
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from pymilvus import (
    connections,
    utility,
    DataType,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)

import recommendation.config as config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==================== Models ====================

class SearchRequest(BaseModel):
    query: str = Field(..., description="用户输入的查询文本")
    longitude: float = Field(..., description="用户当前经度")
    latitude: float = Field(..., description="用户当前纬度")
    pageSize: int = Field(10, description="每页数量")
    pageNum: int = Field(1, description="页码，从1开始")
    sync_type: str = Field("all", description="搜索类型：all/merchant/campus/course/apartment")


class UnifiedRecommendResult(BaseModel):
    id: int
    type: str = Field(..., description="结果类型：merchant/campus/course/apartment")
    name: str
    address: str
    longitude: float
    latitude: float
    distance_km: Optional[float] = None
    description: Optional[str] = ""
    slogen: Optional[str] = ""
    cover_image: Optional[str] = ""
    score: Optional[float] = None
    raw_score: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    price_text: Optional[str] = ""
    reason: Optional[str] = ""


class SearchResponse(BaseModel):
    query: str
    sync_type: str
    total: int
    pageNum: int
    pageSize: int
    query_mode: str = "normal"  # nav / exp_strong / exp_weak
    is_remote: bool = False
    results: List[UnifiedRecommendResult]


# ==================== Embedder ====================

try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba not installed, using simple tokenizer for sparse vector fallback.")


class Embedder:
    def __init__(self):
        self.api_key = config.QWEN_API_KEY
        self.model = config.QWEN_EMBEDDING_MODEL
        self.output_type = config.QWEN_OUTPUT_TYPE
        self.dense_dim = config.DENSE_DIM
        self.sparse_dim = config.SPARSE_DIM
        dashscope.api_key = self.api_key

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        if not texts:
            return [], []
        dense_results = []
        sparse_results = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            dense_batch, sparse_batch = self._call_embedding_api(batch)
            dense_results.extend(dense_batch)
            sparse_results.extend(sparse_batch)
            if i + batch_size < len(texts):
                time.sleep(0.15)
        return dense_results, sparse_results

    def _call_embedding_api(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        dense_list: List[List[float]] = []
        sparse_list: List[Dict[int, float]] = []
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=texts,
                dimension=self.dense_dim,
                output_type=self.output_type,
            )
            if resp.status_code != HTTPStatus.OK:
                logger.error(f"Embedding API error: status_code={resp.status_code}, msg={resp}")
                raise RuntimeError(f"Embedding API failed: {resp}")
            embeddings = resp.output.get("embeddings", [])
            embeddings.sort(key=lambda x: x.get("text_index", 0))
            for emb in embeddings:
                dense = emb.get("embedding", [])
                if not dense:
                    dense = [0.0] * self.dense_dim
                dense_list.append(dense)
                sparse_raw = emb.get("sparse_embedding", [])
                if sparse_raw:
                    sparse = {}
                    for item in sparse_raw:
                        idx = int(item.get("index", 0))
                        val = float(item.get("value", 0.0))
                        if val != 0.0:
                            sparse[idx] = val
                    sparse_list.append(sparse)
                else:
                    sparse_list.append({})
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            for text in texts:
                dense_list.append([0.0] * self.dense_dim)
                sparse_list.append(self._text_to_sparse_vector(text))
        return dense_list, sparse_list

    def _text_to_sparse_vector(self, text: str) -> Dict[int, float]:
        if not text:
            return {}
        tokens = self._tokenize(text)
        bucket_weights: Dict[int, float] = {}
        for token in tokens:
            if len(token) < 2:
                continue
            bucket = self._hash_token(token) % self.sparse_dim
            bucket_weights[bucket] = bucket_weights.get(bucket, 0.0) + 1.0
        return bucket_weights

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        if JIEBA_AVAILABLE:
            return list(jieba.cut(text.strip()))
        else:
            return text.split()

    @staticmethod
    def _hash_token(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def embed_query(self, query: str) -> Tuple[List[float], Dict[int, float]]:
        dense, sparse = self.get_embeddings([query])
        return dense[0] if dense else [0.0] * self.dense_dim, sparse[0] if sparse else {}


# ==================== MilvusClient ====================

class MilvusClient:
    def __init__(self):
        self.host = config.MILVUS_HOST
        self.port = config.MILVUS_PORT
        self.db = config.MILVUS_DB
        self.collection_name = config.MILVUS_COLLECTION
        self.dense_dim = config.DENSE_DIM
        self.sparse_dim = config.SPARSE_DIM
        self.collection: Optional[Collection] = None

    def connect(self):
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            db_name=self.db,
        )
        logger.info(f"Connected to Milvus at {self.host}:{self.port}, db={self.db}")

    def disconnect(self):
        connections.disconnect("default")

    def _load_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"Loaded existing collection: {self.collection_name}")
        else:
            logger.error(f"Collection {self.collection_name} does not exist!")
            raise RuntimeError(f"Collection {self.collection_name} not found")

    def get_collection(self, name: str) -> Collection:
        if utility.has_collection(name):
            coll = Collection(name)
            coll.load()
            return coll
        raise RuntimeError(f"Collection {name} not found")

    def hybrid_search_on_collection(
            self,
            coll: Collection,
            dense_vector: List[float],
            sparse_vector: Dict[int, float],
            top_k: int = 50,
            expr: Optional[str] = None,
    ) -> List[dict]:
        dense_search_params = {
            "data": [dense_vector],
            "anns_field": "dense_vector",
            "param": {"metric_type": "COSINE", "params": {"ef": 64}},
            "limit": top_k,
            "expr": expr,
        }
        dense_req = AnnSearchRequest(**dense_search_params)
        sparse_search_params = {
            "data": [sparse_vector],
            "anns_field": "sparse_vector",
            "param": {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            "limit": top_k,
            "expr": expr,
        }
        sparse_req = AnnSearchRequest(**sparse_search_params)
        ranker = RRFRanker(k=config.RRF_K)
        output_fields = [
            f.name for f in coll.schema.fields
            if f.dtype not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR)
        ]
        results = coll.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=top_k,
            output_fields=output_fields,
        )
        return self._parse_search_results(results)

    @staticmethod
    def _parse_search_results(results) -> List[dict]:
        hits = []
        if not results or len(results) == 0:
            return hits
        for result_group in results:
            for hit in result_group:
                item = {
                    "id": hit.id,
                    "distance": hit.distance,
                }
                if hasattr(hit, "entity") and hit.entity:
                    for key in hit.entity.fields:
                        item[key] = hit.entity.get(key)
                hits.append(item)
        return hits


# ==================== Embedding 内存缓存 ====================

class EmbeddingCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self._cache = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()

    def _make_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Tuple[List[float], Dict[int, float]]]:
        key = self._make_key(text)
        with self._lock:
            if key in self._cache:
                value, expire_at = self._cache[key]
                if time.time() < expire_at:
                    return value
                del self._cache[key]
            return None

    def set(self, text: str, value: Tuple[List[float], Dict[int, float]]):
        key = self._make_key(text)
        with self._lock:
            now = time.time()
            expired = [k for k, (_, exp) in self._cache.items() if exp < now]
            for k in expired:
                del self._cache[k]
            if len(self._cache) >= self._max_size:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (value, time.time() + self._ttl)


# ==================== 轻量级类型意图识别（v2.6） ====================

class TypeIntentRecognizer:
    """
    类型意图识别 + 关键词同义词扩展
    """

    TYPE_KEYWORDS = {
        "merchant": [
            "面包", "蛋糕", "奶茶", "咖啡", "书店", "美食", "小吃", "火锅", "烧烤",
            "酒吧", "清吧", "咖啡馆", "咖啡厅", "奶茶店", "甜品", "面包店", "瑜伽",
            "健身", "画室", "花店", "便利店", "超市", "商场", "影院", "KTV", "餐厅",
            "文具", "自习室", "茶室", "文创", "民宿", "太极", "武术", "烤肉", "日料",
            "韩餐", "西餐", "中餐", "快餐", "饮品", "烘焙", "手作", "杂货", "饰品",
            "服装", "理发", "美容", "按摩", "足疗", "宠物", "摄影", "桌游", "剧本杀"
        ],
        "course": [
            "英语", "瑜伽", "编程", "书法", "绘画", "舞蹈", "吉他", "钢琴", "培训",
            "学习", "课程", "夜校", "素描", "国画", "油画", "声乐", "乐器", "烘焙课",
            "料理课", "手工课", "兴趣", "爱好", "技能", "考证", "考试", "辅导",
            "家教", "教程", "网课", "线下课", "体验课", "试听课", "公开课", "讲座",
            "资格", "成人教育", "继续教育"
        ],
        "campus": [
            "夜校", "青年中心", "自习", "活动空间", "青年之家", "青年宫", "活动中心",
            "社区中心", "党群中心", "新时代文明实践", "文化宫", "图书馆", "阅览室",
            "活动室", "会议室", "多功能厅", "排练厅", "展厅", "报告厅", "礼堂",
            "少年宫", "青少年", "驿站", "服务站"
        ],
        "apartment": [
            "公寓", "租房", "住宿", "青年公寓", "人才公寓", "拎包入住", "合租",
            "整租", "单室套", "一居室", "两居室", "三居室", "宿舍", "床位", "短租",
            "长租", "月付", "季付", "年付", "押一付一", "无中介", "房东直租",
            "精装修", "简装修", "毛坯", "朝南", "带阳台", "独卫", "合租室友",
            "青年驿站", "求职公寓", "过渡房"
        ],
    }

    # 类别词表：包含匹配，query 包含这些词即强制探索型
    CATEGORY_WORDS = {
        "酒吧", "酒馆", "理发", "美发", "面包", "蛋糕", "奶茶", "咖啡", "公寓",
        "课程", "夜校", "火锅", "烧烤", "书店", "瑜伽", "健身", "民宿", "酒店",
        "餐厅", "美食", "甜品", "饮品", "烘焙", "烤肉", "日料", "韩餐", "西餐",
        "中餐", "快餐", "茶室", "自习室", "文具", "文创", "宠物", "摄影", "桌游",
        "剧本杀", "英语", "编程", "书法", "绘画", "舞蹈", "吉他", "钢琴", "培训",
        "学习", "租房", "住宿", "青年公寓", "人才公寓", "拎包入住", "合租", "整租",
        "造型", "剪发", "烫染", "发廊",
    }

    # 关键词同义词映射：用于扩展关键词，提高准入层覆盖率
    KEYWORD_SYNONYMS = {
        "理发": {"理发", "美发", "造型", "剪发", "烫染", "发廊", "护发", "精剪"},
        "酒吧": {"酒吧", "酒馆", "精酿", "调酒", "夜酒", "酒廊", "驻唱", "民谣"},
        "咖啡": {"咖啡", "咖啡馆", "咖啡厅", "拿铁", "美式", "意式", "摩卡", "手冲"},
        "面包": {"面包", "烘焙", "蛋糕", "甜品", "西点", "吐司", "欧包"},
        "奶茶": {"奶茶", "茶饮", "果茶", "奶盖", "珍珠奶茶"},
        "火锅": {"火锅", "涮肉", "串串", "麻辣", "锅底"},
        "烧烤": {"烧烤", "烤肉", "烤串", "撸串", "炭烤"},
        "瑜伽": {"瑜伽", "普拉提", "健身", "塑形", "体态"},
        "英语": {"英语", "外语", "口语", "雅思", "托福", "四六级"},
        "公寓": {"公寓", "租房", "住宿", "青年公寓", "人才公寓", "拎包入住", "合租", "整租", "宿舍"},
        "课程": {"课程", "培训", "学习", "教程", "网课", "线下课", "体验课", "试听课", "公开课", "讲座"},
        "夜校": {"夜校", "青年中心", "自习", "活动空间", "青年之家"},
    }

    def recognize(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        if not query:
            return [], {t: 1.0 for t in self.TYPE_KEYWORDS}

        query_lower = query.lower()
        type_scores = {t: 0 for t in self.TYPE_KEYWORDS}
        matched_keywords = []

        for t, kws in self.TYPE_KEYWORDS.items():
            for kw in kws:
                if kw in query_lower:
                    type_scores[t] += 1
                    matched_keywords.append(kw)

        matched_keywords = list(set(matched_keywords))
        max_score = max(type_scores.values()) if any(type_scores.values()) else 0

        if max_score > 0:
            weights = {}
            for t, score in type_scores.items():
                if score > 0:
                    weights[t] = 1.2 + 0.3 * (score / max_score)
                else:
                    weights[t] = 0.3
        else:
            weights = {t: 1.0 for t in self.TYPE_KEYWORDS}

        return matched_keywords, weights

    def is_category_word(self, query: str) -> bool:
        """类别词包含匹配：query 包含任意类别词即返回 True"""
        if not query:
            return False
        query_norm = query.strip().lower().replace(" ", "").replace("·", "")
        for cw in self.CATEGORY_WORDS:
            if cw in query_norm:
                return True
        return False

    def expand_keywords(self, keywords: List[str]) -> Set[str]:
        """扩展关键词为同义词集合"""
        expanded = set()
        for kw in keywords:
            expanded.add(kw)
            if kw in self.KEYWORD_SYNONYMS:
                expanded.update(self.KEYWORD_SYNONYMS[kw])
        return expanded


# ==================== 规则重排引擎（生产级 v2.6） ====================

class RuleRanker:
    """
    生产级规则重排引擎 v2.6
    - 精确匹配隔离置顶
    - Min-Max 全局归一化
    - 强探索型硬过滤：目标类型未命中扩展关键词直接丢弃
    """

    MODE_WEIGHTS = {
        "nav": (0.60, 0.15, 0.15, 0.10),
        "exp_strong": (0.25, 0.25, 0.40, 0.10),
        "exp_weak": (0.20, 0.30, 0.40, 0.10),
    }

    CV_THRESHOLD = 0.12
    REMOTE_RATIO_THRESHOLD = 0.8
    REMOTE_DISTANCE_KM = 300

    @staticmethod
    def normalize_global_scores(hits: List[dict]) -> List[dict]:
        if not hits:
            return hits
        distances = [h.get("distance", 0) for h in hits]
        min_d = min(distances)
        max_d = max(distances)
        range_d = (max_d - min_d) if max_d != min_d else 1.0

        for h in hits:
            raw = h.get("distance", 0)
            h["norm_score"] = (raw - min_d) / range_d if range_d > 0 else 0.5
        return hits

    @staticmethod
    def keyword_score(hit: dict, keywords: Set[str]) -> float:
        if not keywords:
            return 0.0
        text = f"{hit.get('name', '')} {hit.get('description_text', '')} {hit.get('address', '')}"
        text = text.lower()
        matches = sum(1 for k in keywords if k in text)
        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.25
        elif matches == 2:
            return 0.40
        else:
            return 0.55

    @staticmethod
    def name_match_boost(hit: dict, query: str) -> float:
        if not query:
            return 0.0
        name = hit.get("name", "") or hit.get("course_name", "") or ""
        if not name:
            return 0.0

        query_norm = query.lower().replace(" ", "").replace("·", "").replace("(", "").replace(")", "")
        name_norm = name.lower().replace(" ", "").replace("·", "").replace("(", "").replace(")", "")

        if query_norm in name_norm or name_norm in query_norm:
            return 0.35

        if len(query_norm) >= 3 and len(name_norm) >= 3:
            similarity = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
            if similarity >= 0.85:
                return 0.30
            elif similarity >= 0.70:
                return 0.15
            elif similarity >= 0.55:
                return 0.05

        return 0.0

    @staticmethod
    def distance_score(dist_km: Optional[float]) -> float:
        if dist_km is None or dist_km < 0:
            return 0.0
        if dist_km <= 3:
            return 1.0
        elif dist_km <= 10:
            return 0.9 - (dist_km - 3) * 0.02857
        elif dist_km <= 30:
            return 0.7 - (dist_km - 10) * 0.01
        elif dist_km <= 100:
            return 0.5 - (dist_km - 30) * 0.002857
        elif dist_km <= 300:
            return 0.3 - (dist_km - 100) * 0.001
        else:
            return 0.05

    def apply_relevance_gate(
            self,
            hits: List[dict],
            query: str,
            keywords: List[str],
            type_weights: Dict[str, float],
            sync_type: str,
            type_recognizer: TypeIntentRecognizer,
            query_mode: str,
    ) -> List[dict]:
        """
        相关性准入层 v2.6：
        1. 强探索型（exp_strong）+ 意图明确：目标类型必须命中扩展关键词，否则丢弃
        2. 导航型/弱探索型：保持原逻辑
        3. 非目标类型：语义分 >= 最高 50% 可保留
        """
        if not hits:
            return hits

        # 判断意图明确度
        max_tw = max(type_weights.values())
        dominant_types = [t for t, w in type_weights.items() if w == max_tw]
        is_strong_intent = (max_tw >= 1.5 and sync_type == "all" and
                            sum(1 for w in type_weights.values() if w <= 0.3) >= 3)

        # 扩展关键词
        expanded_kws = type_recognizer.expand_keywords(keywords)
        query_lower = query.lower().replace(" ", "").replace("·", "")
        max_raw = max(h.get("distance", 0) for h in hits)

        filtered = []
        for h in hits:
            t = h.get("type", "merchant")
            name = h.get("name", "") or h.get("course_name", "") or ""
            desc = h.get("description_text", "") or h.get("description", "") or ""
            text = (name + desc).lower()
            raw = h.get("distance", 0)

            # 准入1：名称强匹配（导航型/任何模式都优先保留）
            name_lower = name.lower().replace(" ", "").replace("·", "")
            if query_lower in name_lower or name_lower in query_lower:
                filtered.append(h)
                continue
            if len(query_lower) >= 3 and len(name_lower) >= 3:
                sim = difflib.SequenceMatcher(None, query_lower, name_lower).ratio()
                if sim >= 0.70:
                    filtered.append(h)
                    continue

            # 准入2：强探索型 + 强意图 + 目标类型：必须命中扩展关键词
            if query_mode == "exp_strong" and is_strong_intent and t in dominant_types:
                if any(ek in text for ek in expanded_kws):
                    filtered.append(h)
                    continue
                else:
                    # 未命中任何扩展关键词，直接丢弃（无论语义分多高）
                    logger.debug(f"[Gate] 强过滤丢弃 id={h.get('id')} name={name} type={t} "
                                 f"原因：未命中扩展关键词 {expanded_kws}")
                    continue

            # 准入3：非目标类型或弱意图：语义分阈值
            if is_strong_intent and t not in dominant_types:
                if max_raw > 0 and raw >= max_raw * 0.50:
                    filtered.append(h)
                    continue
            else:
                # 弱意图或单表查询：语义分 >= 最高 25% 保留
                if max_raw > 0 and raw >= max_raw * 0.25:
                    filtered.append(h)
                    continue

            # 未通过准入
            logger.debug(f"[Gate] 丢弃 id={h.get('id')} name={name} type={t} raw={raw:.4f}")

        # 保底
        if not filtered and hits:
            logger.warning(f"[Gate] 全部过滤，启用保底，返回 Top-{min(20, len(hits))}")
            filtered = sorted(hits, key=lambda x: x.get("distance", 0), reverse=True)[:20]

        if query_mode == "exp_strong":
            logger.info(f"[Gate] 强探索型过滤：{len(hits)} → {len(filtered)} 条，"
                        f"扩展关键词={expanded_kws}")

        return filtered

    def detect_mode(self, query: str, hits: List[dict], type_recognizer: TypeIntentRecognizer) -> str:
        if not query or len(query.strip()) < 2:
            return "exp_weak"

        query_norm = query.strip().lower().replace(" ", "").replace("·", "")

        # 强制类别词走探索型（包含匹配）
        if type_recognizer.is_category_word(query):
            return "exp_strong"

        # 导航型判定
        if len(query_norm) >= 4:
            for h in hits[:30]:
                name = h.get("name", "") or h.get("course_name", "") or ""
                if not name:
                    continue
                name_norm = name.lower().replace(" ", "").replace("·", "")
                if query_norm in name_norm or name_norm in query_norm:
                    return "nav"
                sim = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
                if sim >= 0.85:
                    return "nav"

            if len(hits) >= 5:
                top1 = hits[0].get("distance", 0)
                top5 = hits[4].get("distance", 0)
                if top1 > 0 and (top1 - top5) / top1 > 0.40:
                    return "nav"

        return "exp_weak"

    def rank(
            self,
            hits: List[dict],
            keywords: List[str],
            user_lon: float,
            user_lat: float,
            type_weights: Dict[str, float],
            query: str,
            type_recognizer: TypeIntentRecognizer,
    ) -> Tuple[List[dict], str, bool]:
        """
        完整排序流程：
        1. 相关性准入层（含强探索型硬过滤）
        2. 全局归一化
        3. 模式检测
        4. 精确匹配隔离
        5. 常规打分
        6. 精确匹配超线性置顶
        """
        # 1. 先检测模式（用于准入层判断）
        mode = self.detect_mode(query, hits, type_recognizer)

        # 2. 相关性准入层
        hits = self.apply_relevance_gate(
            hits, query, keywords, type_weights, "all", type_recognizer, mode
        )

        # 3. 全局归一化
        hits = self.normalize_global_scores(hits)

        # 4. 异地模式
        far_count = sum(1 for h in hits if h.get("distance_km", 0) > self.REMOTE_DISTANCE_KM)
        remote_mode = len(hits) > 0 and (far_count / len(hits)) >= self.REMOTE_RATIO_THRESHOLD

        w_sem, w_kw, w_dist, w_base = self.MODE_WEIGHTS[mode]
        logger.info(f"[Ranker] 查询模式=[{mode}], 权重=({w_sem}, {w_kw}, {w_dist}, {w_base})")

        # 5. 分离精确匹配
        exact_matches = []
        normal_hits = []

        for h in hits:
            boost = self.name_match_boost(h, query)
            if boost >= 0.30:
                h["_is_exact"] = True
                h["_name_boost"] = boost
                exact_matches.append(h)
            else:
                h["_is_exact"] = False
                h["_name_boost"] = 0
                normal_hits.append(h)

        # 6. 常规结果打分
        expanded_kws = type_recognizer.expand_keywords(keywords)
        for h in normal_hits:
            t = h.get("type", "merchant")
            norm = h.get("norm_score", 0.5)
            kw = self.keyword_score(h, expanded_kws)
            dist = self.distance_score(h.get("distance_km"))
            name_boost = h.get("_name_boost", 0)

            penalty = h.get("_type_weight_penalty", 1.0)
            tw = type_weights.get(t, 1.0) * penalty

            if remote_mode:
                effective_semantic = min(norm + name_boost * 0.5, 1.0)
                base = (0.50 * effective_semantic +
                        0.30 * kw +
                        0.00 * dist +
                        0.20)
            else:
                effective_semantic = min(norm + name_boost * 0.5, 1.0)
                base = (w_sem * effective_semantic +
                        w_kw * kw +
                        w_dist * dist +
                        w_base)

            h["final_score"] = round(base * tw, 4)

        normal_hits.sort(key=lambda x: x["final_score"], reverse=True)

        # 7. 精确匹配超线性置顶
        for h in exact_matches:
            t = h.get("type", "merchant")
            norm = h.get("norm_score", 0.5)
            kw = self.keyword_score(h, expanded_kws)
            dist = self.distance_score(h.get("distance_km"))
            boost = h.get("_name_boost", 0)

            penalty = h.get("_type_weight_penalty", 1.0)
            tw = type_weights.get(t, 1.0) * penalty

            effective_semantic = min(norm + boost, 1.0)
            base = (0.70 * effective_semantic +
                    0.15 * kw +
                    0.05 * dist +
                    0.10)

            raw_score = base * tw
            h["final_score"] = round(raw_score * 1.5 + 0.5, 4)

        exact_matches.sort(key=lambda x: x["final_score"], reverse=True)

        # 8. 最终队列
        final_ranked = exact_matches + normal_hits

        if exact_matches:
            logger.info(f"[Ranker] 精确匹配 {len(exact_matches)} 条置顶: "
                        f"{[h.get('name', '') for h in exact_matches[:3]]}")

        return final_ranked, mode, remote_mode


# ==================== SearchService ====================

class SearchService:
    def __init__(self, milvus_client: MilvusClient, embedder: Embedder):
        self.milvus = milvus_client
        self.embedder = embedder
        self.type_recognizer = TypeIntentRecognizer()
        self.rule_ranker = RuleRanker()
        self.emb_cache = EmbeddingCache(ttl_seconds=300, max_size=1000)
        self.collection_map = {
            "merchant": config.MILVUS_COLLECTION,
            "campus": config.MILVUS_COLLECTION + "_campus",
            "course": config.MILVUS_COLLECTION + "_course",
            "apartment": config.MILVUS_COLLECTION + "_apartment",
        }

    @staticmethod
    def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _search_single(self, sync_type: str, dense_vec: List[float], sparse_vec: Dict[int, float], top_k: int) -> List[
        dict]:
        coll_name = self.collection_map.get(sync_type)
        if not coll_name:
            return []
        try:
            coll = self.milvus.get_collection(coll_name)
            conditions = ["is_use == 1", "status == 1"]
            if sync_type != "course":
                conditions.append("is_open == 1")
            expr = " and ".join(conditions)

            results = self.milvus.hybrid_search_on_collection(
                coll=coll,
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                top_k=top_k,
                expr=expr,
            )
            for r in results:
                r["type"] = sync_type
            return results
        except Exception as e:
            logger.warning(f"[Search] {sync_type} 查询失败: {e}")
            return []

    def _search_all(self, dense_vec: List[float], sparse_vec: Dict[int, float], top_k: int) -> List[dict]:
        all_hits = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._search_single, st, dense_vec, sparse_vec, top_k): st
                for st in self.collection_map.keys()
            }
            for future in as_completed(futures):
                st = futures[future]
                hits = future.result()
                all_hits.extend(hits)
                print(f"[All] {st} 召回 {len(hits)} 条")
        return all_hits

    def _to_unified_result(self, hit: dict) -> UnifiedRecommendResult:
        t = hit.get("type", "merchant")
        name = hit.get("name", "") or hit.get("course_name", "") or hit.get("title", "")
        address = hit.get("address", "") or ""
        desc = hit.get("description_text", "") or hit.get("description", "") or hit.get("intro", "") or ""
        slogen = hit.get("slogen", "") if t == "merchant" else ""
        cover = hit.get("cover_image", "") or hit.get("logo", "") or hit.get("image", "") or ""

        tags = []
        ba = hit.get("business_area", "")
        if ba:
            tags.append(ba)
        cat_id = hit.get("category_id", 0)
        if cat_id:
            tags.append(str(cat_id))

        reason = ""
        dist = hit.get("distance_km")
        score = hit.get("final_score", 0)
        is_exact = hit.get("_is_exact", False)

        if is_exact:
            reason = "精准匹配"
        elif dist is not None and dist <= 3:
            reason = "距离最近"
        elif dist is not None and dist <= 10:
            reason = "附近推荐"
        elif t == "apartment":
            reason = "青年公寓"
        elif t == "course":
            reason = "热门课程"
        elif t == "campus":
            reason = "青年夜校"
        elif t == "merchant":
            reason = "人气小店"

        return UnifiedRecommendResult(
            id=hit.get("id", 0),
            type=t,
            name=name,
            address=address,
            longitude=float(hit.get("longitude", 0)),
            latitude=float(hit.get("latitude", 0)),
            distance_km=round(dist, 2) if dist is not None else None,
            description=desc,
            slogen=slogen,
            cover_image=cover,
            score=round(score, 2),
            raw_score=round(hit.get("distance", 0), 4),
            tags=tags,
            reason=reason,
        )

    def search(self, req: SearchRequest) -> SearchResponse:
        start_total = time.time()
        query = req.query.strip()
        page_size = max(req.pageSize, 1)
        page_num = max(req.pageNum, 1)

        # 1. 意图识别
        t_start = time.time()
        keywords, type_weights = self.type_recognizer.recognize(query)
        print(f"[Search] 查询='{query}', 关键词={keywords}, 类型权重={type_weights}")
        print(f"[Search] 意图识别耗时: {(time.time() - t_start) * 1000:.1f}ms")

        # 2. Embedding
        t_start = time.time()
        cached = self.emb_cache.get(query)
        if cached:
            dense_vec, sparse_vec = cached
            print("[Search] Embedding 命中缓存")
        else:
            dense_vec, sparse_vec = self.embedder.embed_query(query)
            self.emb_cache.set(query, (dense_vec, sparse_vec))

        is_zero_dense = not any(dense_vec)
        if is_zero_dense:
            logger.error("[Search] Embedding 返回零向量，触发服务降级")
            return SearchResponse(
                query=query, sync_type=req.sync_type, total=0,
                pageNum=page_num, pageSize=page_size,
                query_mode="exp_weak", is_remote=False, results=[]
            )

        print(f"[Search] Embedding 耗时: {(time.time() - t_start) * 1000:.1f}ms")

        # 3. 召回
        recall_k = 50 if page_num <= 5 else 100

        # 4. 查询
        t_start = time.time()
        if req.sync_type == "all":
            all_hits = self._search_all(dense_vec, sparse_vec, recall_k)
        else:
            all_hits = self._search_single(req.sync_type, dense_vec, sparse_vec, recall_k)

        for h in all_hits:
            try:
                lon = float(h.get("longitude", 0))
                lat = float(h.get("latitude", 0))
                h["distance_km"] = self.haversine_distance(req.longitude, req.latitude, lon, lat)
            except Exception:
                h["distance_km"] = 9999.0

        print(f"[Search] 查询总耗时: {(time.time() - t_start) * 1000:.1f}ms, 总候选 {len(all_hits)} 条")

        if not all_hits:
            return SearchResponse(
                query=query, sync_type=req.sync_type, total=0,
                pageNum=page_num, pageSize=page_size,
                query_mode="exp_weak", is_remote=False, results=[]
            )

        # 5. 强意图软过滤（all 模式下）
        if req.sync_type == "all":
            max_weight = max(type_weights.values())
            dominant_types = [t for t, w in type_weights.items() if w == max_weight]
            other_weights = [w for t, w in type_weights.items() if t not in dominant_types]
            if max_weight >= 1.5 and all(w <= 0.3 for w in other_weights):
                dominant_type = dominant_types[0]
                logger.info(f"[Search] 强意图软过滤: 主类型=[{dominant_type}]")
                for h in all_hits:
                    if h.get("type") != dominant_type:
                        h["_type_weight_penalty"] = 0.1

        # 6. 规则重排（含准入层 + 精确匹配 + Min-Max 归一化）
        t_start = time.time()
        ranked, mode, is_remote = self.rule_ranker.rank(
            all_hits, keywords, req.longitude, req.latitude, type_weights, query, self.type_recognizer
        )
        print(f"[Search] 规则重排耗时: {(time.time() - t_start) * 1000:.1f}ms, "
              f"模式={mode}, 异地={is_remote}, 过滤后候选={len(ranked)}")

        # 7. 分页
        total = len(ranked)
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        page_hits = ranked[start_idx:end_idx]

        # 8. 映射
        results = [self._to_unified_result(h) for h in page_hits]

        print(f"[Search] 总耗时: {(time.time() - start_total) * 1000:.1f}ms | 返回 {len(results)}/{total} 条")
        return SearchResponse(
            query=query,
            sync_type=req.sync_type,
            total=total,
            pageNum=page_num,
            pageSize=page_size,
            query_mode=mode,
            is_remote=is_remote,
            results=results,
        )


# ==================== FastAPI App ====================

milvus_client: MilvusClient = None
embedder: Embedder = None
search_service: SearchService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global milvus_client, embedder, search_service
    logger.info("[SearchService] Initializing...")
    milvus_client = MilvusClient()
    milvus_client.connect()
    milvus_client._load_collection()
    embedder = Embedder()
    search_service = SearchService(milvus_client, embedder)

    for st, name in search_service.collection_map.items():
        try:
            coll = milvus_client.get_collection(name)
            logger.info(f"Pre-loaded collection for [{st}]: {name}")
        except Exception as e:
            logger.warning(f"Failed to pre-load collection [{st}] {name}: {e}")

    logger.info("[SearchService] Ready.")
    yield
    logger.info("[SearchService] Shutting down...")
    if milvus_client:
        milvus_client.disconnect()
    logger.info("[SearchService] Stopped.")


app = FastAPI(
    title="首页智能搜索服务",
    description="基于 Milvus 向量检索的统一搜索服务（小店/夜校/课程/公寓）",
    version="2.6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", summary="健康检查", tags=["基础"])
def health_check():
    return {
        "service": "home-search-service",
        "status": "ok",
        "milvus_connected": milvus_client is not None,
    }


@app.get("/api/v1", summary="接口列表", tags=["基础"])
def api_index():
    return {
        "service": "首页智能搜索服务",
        "version": "2.6.0",
        "docs": "/docs",
        "endpoints": {
            "health": {"method": "GET", "path": "/health", "desc": "健康检查"},
            "search": {"method": "POST", "path": "/api/v1/search",
                       "desc": "统一智能搜索（all/merchant/campus/course/apartment）"},
        }
    }


@app.post("/api/v1/search", response_model=SearchResponse, summary="统一智能搜索", tags=["搜索"])
def search(req: SearchRequest):
    """
    首页统一搜索入口（生产级 v2.6）：
    - 类别词包含匹配：理发店/酒吧/咖啡店等强制探索型
    - 关键词同义词扩展：理发→{理发,美发,造型,剪发,烫染...}
    - 强探索型硬过滤：目标类型未命中扩展关键词直接丢弃（防咖啡店混入理发店结果）
    - 精确匹配置顶 + Min-Max 归一化
    """
    try:
        print(f"[Search] Request: {req}")
        resp = search_service.search(req)
        return resp
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("home_search_w:app", host="0.0.0.0", port=8033, reload=False)