"""
数据同步服务 API
独立部署，负责从达梦数据库同步数据到 Milvus
"""
import csv
import hashlib
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple

import dashscope
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==================== Models ====================

class Merchant(BaseModel):
    id: int
    register_id: int
    name: str
    logo: Optional[str] = ""
    cover_image: Optional[str] = ""
    category_id: int = 0
    address: str
    city_code: str
    district_code: Optional[str] = ""
    business_area: Optional[str] = ""
    longitude: float
    latitude: float
    contact_name: Optional[str] = ""
    contact_phone: Optional[str] = ""
    slogen: Optional[str] = ""
    description: Optional[str] = ""
    content: Optional[str] = ""
    is_open: int = 1
    is_use: int = 1
    status: int = 1
    digitalization: Optional[str] = ""
    created_at: Optional[str] = ""
    updated_at: Optional[str] = ""
    deleted_at: Optional[str] = None
    creator_id: Optional[int] = 0
    province_code: Optional[str] = "320000"


class MerchantVector(BaseModel):
    id: int
    dense_vector: List[float]
    sparse_vector: dict
    name: str
    address: str
    city_code: str
    district_code: str
    business_area: str
    longitude: float
    latitude: float
    category_id: int
    is_open: int
    is_use: int
    status: int
    description_text: str
    updated_at: int


class RecommendRequest(BaseModel):
    query: str = Field(..., description="用户输入的查询文本")
    longitude: float = Field(..., description="用户当前经度")
    latitude: float = Field(..., description="用户当前纬度")
    pageSize: int = Field(10, description="每页数量")
    pageNum: int = Field(1, description="页码，从1开始")


class RecommendResult(BaseModel):
    id: int
    name: str
    address: str
    business_area: Optional[str] = ""
    longitude: float
    latitude: float
    distance_km: Optional[float] = None
    description: Optional[str] = ""
    slogen: Optional[str] = ""
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    reason: Optional[str] = ""


class RecommendResponse(BaseModel):
    query: str
    total: int
    results: List[RecommendResult]
    is_statistical: bool = False
    statistical_answer: Optional[str] = None


class SyncResponse(BaseModel):
    success: bool
    message: str
    synced_count: int
    deleted_count: int


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
        self.enable_llm_rewrite = bool(getattr(config, "SEARCH_REWRITE_ENABLE", False))
        self.rewrite_model = getattr(config, "SEARCH_REWRITE_MODEL", "qwen-turbo")

    @staticmethod
    def _normalize_description(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[!！。]{2,}", "。", text)
        return text[:120]

    @classmethod
    def _rewrite_to_search_brief(cls, name: str, raw_desc: str, entity_type: str) -> str:
        normalized = cls._normalize_description(raw_desc)
        if not normalized:
            return ""
        stripped = re.sub(r"[\W_]+", "", normalized)
        # 描述噪音较高时，退化为稳定的“名称+类型”短文本
        if len(stripped) < 6:
            return f"{name}，{entity_type}" if name else entity_type
        return normalized


    def _rewrite_with_llm(self, name: str, raw_desc: str, entity_type: str) -> str:
        if not self.enable_llm_rewrite:
            return ""
        prompt = (
            "你是本地生活搜索优化助手。请把下面信息改写成15~30字中文短句，仅保留实体类型与主营内容，不要地址、营销词。\n"
            f"名称：{name}\n类型：{entity_type}\n原描述：{raw_desc}\n"
            "只输出改写结果。"
        )
        try:
            resp = dashscope.Generation.call(
                model=self.rewrite_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=80,
                result_format="message",
            )
            if resp.status_code == HTTPStatus.OK:
                content = ((resp.output or {}).get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                return self._normalize_description(content)
        except Exception as e:
            logger.warning(f"LLM rewrite failed, fallback to rule-based text: {e}")
        return ""

    def build_description_text(self, merchant: dict) -> str:
        name = (merchant.get("name", "") or "").strip()
        category = (merchant.get("category_name", "") or "").strip()
        if not category:
            cid = merchant.get("category_id")
            category = f"小店分类{cid}" if cid not in (None, "") else "小店"
        brief = self._rewrite_with_llm(name, merchant.get("description", ""), category)
        if not brief:
            brief = self._rewrite_to_search_brief(name, merchant.get("description", ""), category)
        parts = [f"店铺名称：{name}", f"店铺类型：{category}"]
        if brief:
            parts.append(f"店铺简介：{brief}")
        return "\n".join([p for p in parts if p and not p.endswith("：")])

    def build_campus_text(self, campus: dict) -> str:
        name = (campus.get("name", "") or "").strip()
        brief = self._rewrite_with_llm(name, campus.get("service_summary", ""), "夜校")
        if not brief:
            brief = self._rewrite_to_search_brief(name, campus.get("service_summary", ""), "夜校")
        parts = [f"夜校名称：{name}", "类型：夜校"]
        if brief:
            parts.append(f"简介：{brief}")
        return "\n".join([p for p in parts if p])

    def build_course_text(self, course: dict) -> str:
        name = (course.get("name", "") or "").strip()
        brief = self._rewrite_with_llm(name, course.get("description", ""), "课程")
        if not brief:
            brief = self._rewrite_to_search_brief(name, course.get("description", ""), "课程")
        parts = [f"课程名称：{name}", "类型：课程"]
        if brief:
            parts.append(f"简介：{brief}")
        return "\n".join([p for p in parts if p])

    def build_apartment_text(self, apartment: dict) -> str:
        name = (apartment.get("name", "") or "").strip()
        brief = self._rewrite_with_llm(name, apartment.get("description", ""), "公寓")
        if not brief:
            brief = self._rewrite_to_search_brief(name, apartment.get("description", ""), "公寓")
        parts = [f"公寓名称：{name}", "类型：公寓"]
        if brief:
            parts.append(f"简介：{brief}")
        return "\n".join([p for p in parts if p])

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

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        logger.info(f"Generating embeddings for {len(texts)} texts via {self.model}...")
        return self.get_embeddings(texts)

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

    def _infer_sync_type(self) -> str:
        """根据 collection_name 后缀推断 sync_type"""
        name = self.collection_name
        if name.endswith("_course"):
            return "course"
        elif name.endswith("_campus"):
            return "campus"
        elif name.endswith("_apartment"):
            return "apartment"
        return "merchant"

    @staticmethod
    def _collection_fields(dense_dim: int, sync_type: str = "merchant") -> List[FieldSchema]:
        """
        按 sync_type 定义 Milvus 集合字段，与达梦 DDL 对齐：
        - 只有 course 包含 campus_id
        - 其余 merchant/campus/apartment 均不含 campus_id
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="address", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="city_code", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="district_code", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="business_area", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="longitude", dtype=DataType.FLOAT),
            FieldSchema(name="latitude", dtype=DataType.FLOAT),
            FieldSchema(name="category_id", dtype=DataType.INT64),
            FieldSchema(name="is_open", dtype=DataType.INT8),
            FieldSchema(name="is_use", dtype=DataType.INT8),
            FieldSchema(name="status", dtype=DataType.INT8),
            FieldSchema(name="description_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="updated_at", dtype=DataType.INT64),
        ]
        # 仅 course 表需要 campus_id（与达梦 DDL 对齐）
        if sync_type == "course":
            fields.insert(1, FieldSchema(name="campus_id", dtype=DataType.INT64))
        return fields

    def create_collection(self, drop_existing: bool = False):
        sync_type = self._infer_sync_type()
        target_fields = self._collection_fields(self.dense_dim, sync_type)

        if drop_existing and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")

        if utility.has_collection(self.collection_name):
            existing = Collection(self.collection_name)
            if self._needs_recreate(existing, sync_type):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection due to schema mismatch: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists and schema matches.")
                self.collection = Collection(self.collection_name)
                return

        schema = CollectionSchema(target_fields, description="Vector collection for recommendation")
        self.collection = Collection(self.collection_name, schema)
        self._create_indexes()
        logger.info(f"Created collection: {self.collection_name} (sync_type={sync_type}, fields={len(target_fields)})")

    def _create_indexes(self):
        dense_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        self.collection.create_index("dense_vector", dense_index_params)
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2},
        }
        self.collection.create_index("sparse_vector", sparse_index_params)
        self.collection.load()
        logger.info("Indexes created and collection loaded.")

    def _needs_recreate(self, existing: Collection, sync_type: str = None) -> bool:
        if sync_type is None:
            sync_type = self._infer_sync_type()
        target_fields = self._collection_fields(self.dense_dim, sync_type)
        existing_names = {f.name for f in existing.schema.fields}
        target_names = {f.name for f in target_fields}

        # 字段缺失或多余都触发重建
        if target_names != existing_names:
            missing = target_names - existing_names
            extra = existing_names - target_names
            logger.warning(
                f"Schema mismatch for {sync_type}: missing={missing}, extra={extra}. "
                f"Will recreate collection."
            )
            return True

        # 检查 VARCHAR 长度
        for ef in existing.schema.fields:
            if ef.dtype == DataType.VARCHAR:
                for tf in target_fields:
                    if tf.name == ef.name and tf.dtype == DataType.VARCHAR:
                        existing_max = ef.params.get("max_length", 0)
                        needed_max = tf.params.get("max_length", 0)
                        if existing_max < needed_max:
                            logger.warning(
                                f"Schema mismatch: field '{ef.name}' existing max_length={existing_max} "
                                f"but needed={needed_max}. Will recreate collection."
                            )
                            return True
        return False

    def ensure_collection(self):
        if utility.has_collection(self.collection_name):
            existing = Collection(self.collection_name)
            sync_type = self._infer_sync_type()
            if self._needs_recreate(existing, sync_type):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection due to schema mismatch: {self.collection_name}")
                self.collection = None
                self.create_collection()
            else:
                if not self.collection:
                    self.collection = existing
                    self.collection.load()
        else:
            self.create_collection()

    def upsert_merchants(self, records: List[dict]):
        self.ensure_collection()
        if not records:
            return

        # 按当前 schema 字段顺序自动构建数据，彻底避免字段数量不匹配
        schema_fields = [f.name for f in self.collection.schema.fields]
        entities = []
        for field_name in schema_fields:
            col_data = [r.get(field_name) for r in records]
            entities.append(col_data)

        self.collection.upsert(entities)
        logger.info(f"Upserted {len(records)} records into {self.collection_name}.")

    def delete_by_ids(self, ids: List[int]):
        self.ensure_collection()
        if not ids:
            return
        expr = f"id in {ids}"
        self.collection.delete(expr)
        logger.info(f"Deleted {len(ids)} records from {self.collection_name}.")

    def delete_by_expr(self, expr: str):
        self.ensure_collection()
        self.collection.delete(expr)
        logger.info(f"Deleted records by expr: {expr}")

    def hybrid_search(
            self,
            dense_vector: List[float],
            sparse_vector: Dict[int, float],
            top_k: int = 50,
            expr: Optional[str] = None,
    ) -> List[dict]:
        self.ensure_collection()
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

        # 自动获取当前集合的所有标量字段，避免查询不存在的字段
        output_fields = [
            f.name for f in self.collection.schema.fields
            if f.dtype not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR)
        ]

        results = self.collection.hybrid_search(
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

    def query(self, expr: str, output_fields: List[str], limit: int = 1000, offset: int = 0) -> List[dict]:
        self.ensure_collection()
        return self.collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit,
            offset=offset,
        )

    def get_all_ids(self) -> List[int]:
        self.ensure_collection()
        ids = []
        batch_size = 1000
        offset = 0
        while True:
            batch = self.collection.query(
                expr="id >= 0",
                output_fields=["id"],
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break
            ids.extend([r["id"] for r in batch])
            if len(batch) < batch_size:
                break
            offset += batch_size
        return ids


# ==================== SyncService ====================

class DataSource:
    def fetch_all(self) -> List[dict]:
        raise NotImplementedError

    def fetch_incremental(self, since: Optional[datetime] = None) -> Tuple[List[dict], List[int]]:
        raise NotImplementedError


class BaseDamengDataSource(DataSource):
    def __init__(self, host: str, port: int, user: str, password: str, database: str = ""):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def _get_connection(self):
        try:
            import dmPython
        except ImportError:
            logger.error("dmPython not installed. Cannot connect to Dameng DB.")
            return None
        return dmPython.connect(
            user=self.user,
            password=self.password,
            server=self.host,
            port=self.port,
        )


class DamengDataSource(BaseDamengDataSource):
    @staticmethod
    def _parse_row(columns: List[str], row: tuple) -> dict:
        r = dict(zip(columns, row))
        r["id"] = int(r["id"]) if r["id"] is not None else 0
        r["longitude"] = float(r["longitude"]) if r["longitude"] is not None else 0.0
        r["latitude"] = float(r["latitude"]) if r["latitude"] is not None else 0.0
        r["category_id"] = int(r["category_id"]) if r["category_id"] is not None else 0
        r["is_open"] = int(r["is_open"]) if r["is_open"] is not None else 1
        r["is_use"] = int(r["is_use"]) if r["is_use"] is not None else 1
        r["status"] = int(r["status"]) if r["status"] is not None else 1
        return r

    def _build_base_sql(self) -> str:
        return """
            SELECT id, register_id, name, logo, cover_image, category_id,
                   address, city_code, district_code, business_area,
                   longitude, latitude, contact_name, contact_phone,
                   slogen, description, content, is_open, is_use, status,
                   digitalization, created_at, updated_at, deleted_at,
                   creator_id, province_code
            FROM merchant_main
        """

    def fetch_all(self) -> List[dict]:
        conn = self._get_connection()
        if not conn:
            return []
        cursor = conn.cursor()
        sql = self._build_base_sql() + " WHERE deleted_at IS NULL"
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        records = [self._parse_row(columns, row) for row in rows]
        cursor.close()
        conn.close()
        logger.info(f"Full fetch: {len(records)} records.")
        return records

    def fetch_incremental(self, since: Optional[datetime] = None) -> Tuple[List[dict], List[int]]:
        conn = self._get_connection()
        if not conn:
            return [], []
        cursor = conn.cursor()
        records = []
        deleted_ids = []
        if since:
            update_sql = self._build_base_sql() + " WHERE deleted_at IS NULL AND updated_at > ?"
            cursor.execute(update_sql, (since,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            delete_sql = "SELECT id FROM merchant_main WHERE deleted_at IS NOT NULL AND updated_at > ?"
            cursor.execute(delete_sql, (since,))
            deleted_ids = [int(row[0]) for row in cursor.fetchall()]
            logger.info(
                f"Incremental fetch: {len(records)} updated, {len(deleted_ids)} deleted since {since}"
            )
        else:
            sql = self._build_base_sql() + " WHERE deleted_at IS NULL"
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            logger.info(f"Incremental fetch (fallback to full): {len(records)} records.")
        cursor.close()
        conn.close()
        return records, deleted_ids


class CampusDataSource(BaseDamengDataSource):
    @staticmethod
    def _parse_row(columns: List[str], row: tuple) -> dict:
        r = dict(zip(columns, row))
        r["id"] = int(r["id"]) if r["id"] is not None else 0
        r["longitude"] = float(r["longitude"]) if r["longitude"] is not None else 0.0
        r["latitude"] = float(r["latitude"]) if r["latitude"] is not None else 0.0
        r["status"] = int(r["status"]) if r["status"] is not None else 1
        r["is_use"] = int(r["is_use"]) if r["is_use"] is not None else 1
        return r

    def _build_base_sql(self) -> str:
        return """
            SELECT id, campus_register_id, project_type, campus_type, name,
                   logo, cover_image, province_code, city_code, district_code,
                   address, business_area, latitude, longitude,
                   service_summary, service_target, contact_name, contact_phone,
                   status, is_use, digitalization, creator_id,
                   created_at, updated_at, deleted_at, jump_type, jump_url
            FROM campus
        """

    def fetch_all(self) -> List[dict]:
        conn = self._get_connection()
        if not conn:
            return []
        cursor = conn.cursor()
        sql = self._build_base_sql() + " WHERE deleted_at IS NULL"
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        records = [self._parse_row(columns, row) for row in rows]
        cursor.close()
        conn.close()
        logger.info(f"Campus full fetch: {len(records)} records.")
        return records

    def fetch_incremental(self, since: Optional[datetime] = None) -> Tuple[List[dict], List[int]]:
        conn = self._get_connection()
        if not conn:
            return [], []
        cursor = conn.cursor()
        records = []
        deleted_ids = []
        if since:
            update_sql = self._build_base_sql() + " WHERE deleted_at IS NULL AND updated_at > ?"
            cursor.execute(update_sql, (since,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            delete_sql = "SELECT id FROM campus WHERE deleted_at IS NOT NULL AND updated_at > ?"
            cursor.execute(delete_sql, (since,))
            deleted_ids = [int(row[0]) for row in cursor.fetchall()]
            logger.info(
                f"Campus incremental fetch: {len(records)} updated, {len(deleted_ids)} deleted since {since}"
            )
        else:
            sql = self._build_base_sql() + " WHERE deleted_at IS NULL"
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            logger.info(f"Campus incremental fetch (fallback to full): {len(records)} records.")
        cursor.close()
        conn.close()
        return records, deleted_ids


class CourseDataSource(BaseDamengDataSource):
    @staticmethod
    def _parse_row(columns: List[str], row: tuple) -> dict:
        r = dict(zip(columns, row))
        r["id"] = int(r["id"]) if r["id"] is not None else 0
        r["status"] = int(r["status"]) if r["status"] is not None else 1
        r["is_use"] = int(r["is_use"]) if r["is_use"] is not None else 1
        r["category_levle1_id"] = int(r["category_levle1_id"]) if r["category_levle1_id"] is not None else 0
        r["longitude"] = float(r["longitude"]) if r["longitude"] is not None else 0.0
        r["latitude"] = float(r["latitude"]) if r["latitude"] is not None else 0.0
        return r

    def _build_base_sql(self) -> str:
        return """
            SELECT
                c.id, c.campus_id, c.name, c.cover_image, c.description,
                c.category_levle1_id, c.category_level2_id, c.teaching_form,
                c.max_students, c.enrolled_count, c.total_lessons, c.lessons_time,
                c.origin_price, c.discount_price, c.price_remark, c.sex_limit,
                c.status, c.open_condition, c.remark, c.is_use, c.start_date, c.end_date,
                c.creator_id, c.created_at, c.updated_at, c.registration_status,
                c.jump_type, c.jump_url, c.registration_start_time, c.registration_end_time,
                c.group_qrcode, c.contact_name, c.contact_phone, c.price_setting,
                cp.address, cp.city_code, cp.district_code, cp.business_area,
                cp.longitude, cp.latitude
            FROM course c
            LEFT JOIN campus cp ON c.campus_id = cp.id
        """

    def fetch_all(self) -> List[dict]:
        conn = self._get_connection()
        if not conn:
            return []
        cursor = conn.cursor()
        sql = self._build_base_sql()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        records = [self._parse_row(columns, row) for row in rows]
        cursor.close()
        conn.close()
        logger.info(f"Course full fetch: {len(records)} records.")
        return records

    def fetch_incremental(self, since: Optional[datetime] = None) -> Tuple[List[dict], List[int]]:
        conn = self._get_connection()
        if not conn:
            return [], []
        cursor = conn.cursor()
        records = []
        if since:
            update_sql = self._build_base_sql() + " WHERE updated_at > ?"
            cursor.execute(update_sql, (since,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            logger.info(
                f"Course incremental fetch: {len(records)} updated since {since}"
            )
        else:
            sql = self._build_base_sql()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            logger.info(f"Course incremental fetch (fallback to full): {len(records)} records.")
        cursor.close()
        conn.close()
        return records, []


class ApartmentDataSource(BaseDamengDataSource):
    @staticmethod
    def _parse_row(columns: List[str], row: tuple) -> dict:
        r = dict(zip(columns, row))
        r["id"] = int(r["id"]) if r["id"] is not None else 0
        r["longitude"] = float(r["longitude"]) if r["longitude"] is not None else 0.0
        r["latitude"] = float(r["latitude"]) if r["latitude"] is not None else 0.0
        r["category_id"] = int(r["category_id"]) if r["category_id"] is not None else 0
        r["is_open"] = int(r["is_open"]) if r["is_open"] is not None else 1
        r["is_use"] = int(r["is_use"]) if r["is_use"] is not None else 1
        r["status"] = int(r["status"]) if r["status"] is not None else 1
        return r

    def _build_base_sql(self) -> str:
        return """
            SELECT id, register_id, name, category_id, address, province_code,
                   city_code, district_code, longitude, latitude, contact_name,
                   description, traffic_info, welfare_switch, welfare_info,
                   is_open, status, is_use, creator_id, created_at, updated_at,
                   deleted_at, responsible_person, slogan
            FROM apartment_main
        """

    def fetch_all(self) -> List[dict]:
        conn = self._get_connection()
        if not conn:
            return []
        cursor = conn.cursor()
        sql = self._build_base_sql() + " WHERE deleted_at IS NULL"
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        records = [self._parse_row(columns, row) for row in rows]
        cursor.close()
        conn.close()
        logger.info(f"Apartment full fetch: {len(records)} records.")
        return records

    def fetch_incremental(self, since: Optional[datetime] = None) -> Tuple[List[dict], List[int]]:
        conn = self._get_connection()
        if not conn:
            return [], []
        cursor = conn.cursor()
        records = []
        deleted_ids = []
        if since:
            update_sql = self._build_base_sql() + " WHERE deleted_at IS NULL AND updated_at > ?"
            cursor.execute(update_sql, (since,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            delete_sql = "SELECT id FROM apartment_main WHERE deleted_at IS NOT NULL AND updated_at > ?"
            cursor.execute(delete_sql, (since,))
            deleted_ids = [int(row[0]) for row in cursor.fetchall()]
            logger.info(
                f"Apartment incremental fetch: {len(records)} updated, {len(deleted_ids)} deleted since {since}"
            )
        else:
            sql = self._build_base_sql() + " WHERE deleted_at IS NULL"
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            records = [self._parse_row(columns, row) for row in rows]
            logger.info(f"Apartment incremental fetch (fallback to full): {len(records)} records.")
        cursor.close()
        conn.close()
        return records, deleted_ids


class SyncService:
    def __init__(
            self,
            milvus_client: MilvusClient,
            embedder: Embedder,
    ):
        self.milvus = milvus_client
        self.embedder = embedder

    @staticmethod
    def _get_data_source(sync_type: str) -> DataSource:
        if sync_type == "campus":
            return CampusDataSource(
                host=config.DM_HOST,
                port=config.DM_PORT,
                user=config.DM_USER,
                password=config.DM_PASSWORD,
                database=config.DM_DATABASE,
            )
        elif sync_type == "course":
            return CourseDataSource(
                host=config.DM_HOST,
                port=config.DM_PORT,
                user=config.DM_USER,
                password=config.DM_PASSWORD,
                database=config.DM_DATABASE,
            )
        elif sync_type == "apartment":
            return ApartmentDataSource(
                host=config.DM_HOST,
                port=config.DM_PORT,
                user=config.DM_USER,
                password=config.DM_PASSWORD,
                database=config.DM_DATABASE,
            )
        else:
            logger.info("Using Dameng DB merchant_main as data source.")
            return DamengDataSource(
                host=config.DM_HOST,
                port=config.DM_PORT,
                user=config.DM_USER,
                password=config.DM_PASSWORD,
                database=config.DM_DATABASE,
            )

    def _switch_collection(self, sync_type: str):
        suffix = "" if sync_type == "merchant" else f"_{sync_type}"
        target = config.MILVUS_COLLECTION + suffix
        if self.milvus.collection_name != target:
            self.milvus.collection_name = target
            self.milvus.collection = None
            self.milvus.ensure_collection()
            logger.info(f"Switched to Milvus collection: {target}")

    def _last_sync_file(self, sync_type: str) -> str:
        return f"{config.LAST_SYNC_TIME_FILE}_{sync_type}"

    def _load_last_sync_time(self, sync_type: str) -> Optional[datetime]:
        path = self._last_sync_file(sync_type)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    return datetime.fromisoformat(content)
            except Exception as e:
                logger.warning(f"Failed to load last_sync_time: {e}")
        return None

    def _save_last_sync_time(self, dt: datetime, sync_type: str):
        path = self._last_sync_file(sync_type)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(dt.isoformat())
        except Exception as e:
            logger.warning(f"Failed to save last_sync_time: {e}")

    def _build_text(self, record: dict, sync_type: str) -> str:
        if sync_type == "campus":
            return self.embedder.build_campus_text(record)
        elif sync_type == "course":
            return self.embedder.build_course_text(record)
        elif sync_type == "apartment":
            return self.embedder.build_apartment_text(record)
        else:
            return self.embedder.build_description_text(record)

    def _to_milvus_record(self, r: dict, text: str, dense_vec: List[float], sparse_vec: dict, sync_type: str) -> dict:
        current_ts = int(datetime.now().timestamp())

        if sync_type == "course":
            return {
                "id": r["id"],
                "campus_id": int(r.get("campus_id", 0)),
                "dense_vector": dense_vec,
                "sparse_vector": sparse_vec,
                "name": r.get("name", ""),
                "address": (r.get("address") or ""),
                "city_code": (r.get("city_code") or ""),
                "district_code": (r.get("district_code") or ""),
                "business_area": (r.get("business_area") or ""),
                "longitude": float(r.get("longitude", 0.0)),
                "latitude": float(r.get("latitude", 0.0)),
                "category_id": int(r.get("category_levle1_id", 0)),
                "is_open": 1,
                "is_use": int(r.get("is_use", 1)),
                "status": int(r.get("status", 1)),
                "description_text": text,
                "updated_at": current_ts,
            }
        elif sync_type == "campus":
            return {
                "id": r["id"],
                "dense_vector": dense_vec,
                "sparse_vector": sparse_vec,
                "name": r.get("name", ""),
                "address": (r.get("address") or ""),
                "city_code": (r.get("city_code") or ""),
                "district_code": (r.get("district_code") or ""),
                "business_area": (r.get("business_area") or ""),
                "longitude": float(r.get("longitude", 0.0)),
                "latitude": float(r.get("latitude", 0.0)),
                "category_id": 0,
                "is_open": int(r.get("status", 1)),
                "is_use": int(r.get("is_use", 1)),
                "status": int(r.get("status", 1)),
                "description_text": text,
                "updated_at": current_ts,
            }
        elif sync_type == "apartment":
            return {
                "id": r["id"],
                "dense_vector": dense_vec,
                "sparse_vector": sparse_vec,
                "name": r.get("name", ""),
                "address": (r.get("address") or ""),
                "city_code": (r.get("city_code") or ""),
                "district_code": (r.get("district_code") or ""),
                "business_area": "",
                "longitude": float(r.get("longitude", 0.0)),
                "latitude": float(r.get("latitude", 0.0)),
                "category_id": int(r.get("category_id", 0)),
                "is_open": int(r.get("is_open", 1)),
                "is_use": int(r.get("is_use", 1)),
                "status": int(r.get("status", 1)),
                "description_text": text,
                "updated_at": current_ts,
            }
        else:  # merchant
            return {
                "id": r["id"],
                "dense_vector": dense_vec,
                "sparse_vector": sparse_vec,
                "name": r.get("name", ""),
                "address": (r.get("address") or ""),
                "city_code": (r.get("city_code") or ""),
                "district_code": (r.get("district_code") or ""),
                "business_area": (r.get("business_area") or ""),
                "longitude": float(r.get("longitude", 0.0)),
                "latitude": float(r.get("latitude", 0.0)),
                "category_id": int(r.get("category_id", 0)),
                "is_open": int(r.get("is_open", 1)),
                "is_use": int(r.get("is_use", 1)),
                "status": int(r.get("status", 1)),
                "description_text": text,
                "updated_at": current_ts,
            }

    def _process_records(self, records: List[dict], sync_type: str) -> List[dict]:
        if not records:
            return []
        valid_records = []
        texts = []
        for r in records:
            if not r.get("name"):
                continue
            text = self._build_text(r, sync_type)
            texts.append(text)
            valid_records.append(r)
        if not valid_records:
            return []
        dense_vectors, sparse_vectors = self.embedder.embed_texts(texts)
        milvus_records = []
        for i, r in enumerate(valid_records):
            milvus_records.append(self._to_milvus_record(
                r, texts[i], dense_vectors[i], sparse_vectors[i], sync_type
            ))
        return milvus_records

    def sync(self, sync_type: str = "merchant") -> Dict:
        start_time = time.time()
        logger.info(f"Starting incremental sync for {sync_type}...")
        sync_start = datetime.now()
        self._switch_collection(sync_type)
        data_source = self._get_data_source(sync_type)
        last_sync_time = self._load_last_sync_time(sync_type)
        records, deleted_ids = data_source.fetch_incremental(last_sync_time)
        if not records and not deleted_ids:
            elapsed = time.time() - start_time
            logger.info(f"No changes since {last_sync_time}. Sync completed in {elapsed:.2f}s.")
            return {
                "success": True,
                "message": "No changes detected",
                "synced_count": 0,
                "deleted_count": 0,
            }
        synced_count = 0
        if records:
            logger.info(f"Processing {len(records)} incremental records...")
            milvus_records = self._process_records(records, sync_type)
            if milvus_records:
                self.milvus.upsert_merchants(milvus_records)
                synced_count = len(milvus_records)
        if deleted_ids:
            self.milvus.delete_by_ids(deleted_ids)
            logger.info(f"Deleted {len(deleted_ids)} soft-deleted records.")
        self._save_last_sync_time(sync_start, sync_type)
        elapsed = time.time() - start_time
        logger.info(f"Incremental sync completed in {elapsed:.2f}s.")
        return {
            "success": True,
            "message": "Incremental sync completed",
            "synced_count": synced_count,
            "deleted_count": len(deleted_ids),
        }

    def sync_full(self, sync_type: str = "merchant") -> Dict:
        start_time = time.time()
        logger.info(f"Starting full sync for {sync_type}...")
        sync_start = datetime.now()
        self._switch_collection(sync_type)
        data_source = self._get_data_source(sync_type)
        records = data_source.fetch_all()
        if not records:
            return {"success": False, "message": "No data fetched", "synced_count": 0, "deleted_count": 0}
        logger.info(f"Fetched {len(records)} records from data source.")
        milvus_records = self._process_records(records, sync_type)
        if milvus_records:
            self.milvus.upsert_merchants(milvus_records)
        existing_ids = set(self.milvus.get_all_ids())
        new_ids = {r["id"] for r in milvus_records}
        ids_to_delete = list(existing_ids - new_ids)
        if ids_to_delete:
            self.milvus.delete_by_ids(ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} obsolete records.")
        self._save_last_sync_time(sync_start, sync_type)
        elapsed = time.time() - start_time
        logger.info(f"Full sync completed in {elapsed:.2f}s.")
        return {
            "success": True,
            "message": "Full sync completed",
            "synced_count": len(milvus_records),
            "deleted_count": len(ids_to_delete),
        }


# ==================== FastAPI App ====================

milvus_client: MilvusClient = None
embedder: Embedder = None
sync_service: SyncService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global milvus_client, embedder, sync_service
    logger.info("[SyncService] Initializing...")
    milvus_client = MilvusClient()
    milvus_client.connect()
    milvus_client.create_collection()
    embedder = Embedder()
    sync_service = SyncService(milvus_client, embedder)
    logger.info("[SyncService] Ready.")
    yield
    logger.info("[SyncService] Shutting down...")
    if milvus_client:
        milvus_client.disconnect()
    logger.info("[SyncService] Stopped.")


app = FastAPI(
    title="数据同步服务",
    description="从达梦数据库同步店铺/夜校/课程/公寓数据到 Milvus 向量数据库",
    version="1.0.0",
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
        "service": "sync-service",
        "status": "ok",
        "milvus_connected": milvus_client is not None,
    }


@app.get("/api/v1", summary="接口列表", tags=["基础"])
def api_index():
    return {
        "service": "数据同步服务",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": {"method": "GET", "path": "/health", "desc": "健康检查"},
            "sync": {"method": "POST", "path": "/api/v1/sync/merchants",
                     "desc": "触发增量数据同步（达梦DB -> Milvus），参数 sync_type=merchant/campus/course/apartment"},
            "sync_full": {"method": "POST", "path": "/api/v1/sync/merchants/full",
                          "desc": "触发全量数据同步（达梦DB -> Milvus），参数 sync_type=merchant/campus/course/apartment"},
        }
    }


@app.post("/api/v1/sync/merchants", response_model=SyncResponse, summary="增量数据同步", tags=["数据同步"])
def sync_merchants(sync_type: str = "merchant"):
    try:
        result = sync_service.sync(sync_type=sync_type)
        return SyncResponse(**result)
    except Exception as e:
        logger.exception("Incremental sync failed")
        raise HTTPException(status_code=500, detail=f"Incremental sync failed: {str(e)}")


@app.post("/api/v1/sync/merchants/full", response_model=SyncResponse, summary="全量数据同步", tags=["数据同步"])
def sync_merchants_full(sync_type: str = "merchant"):
    try:
        result = sync_service.sync_full(sync_type=sync_type)
        return SyncResponse(**result)
    except Exception as e:
        logger.exception("Full sync failed")
        raise HTTPException(status_code=500, detail=f"Full sync failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sync_api:app", host="0.0.0.0", port=8032, reload=True)
