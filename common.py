# -*- coding: utf-8 -*-
"""
公共模块 - 提供共享的类、函数和工厂方法，不维护全局状态
"""
import asyncio
import json
import os
import re
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, Union, List

import numpy as np
import aiohttp
import aiomysql
from fastapi import HTTPException
from pydantic import BaseModel
from pymilvus import AnnSearchRequest, RRFRanker

# 导入配置
from config import (
    EMBEDDING_DIM, VECTOR_DIM,
    DB_CONFIG, DASHSCOPE_API_KEY, EMBEDDING_API_URL, RERANK_API_URL,
    HTTP_TIMEOUT_TOTAL, HTTP_TIMEOUT_CONNECT, HTTP_CONNECTOR_LIMIT, HTTP_CONNECTOR_LIMIT_PER_HOST,
    LOCAL_EMBEDDINGS_CACHE_PATH,
    THRESHOLD,
    RESUME_FIELD_WEIGHTS, ONLINE_RESUME_FIELD_WEIGHTS,
    COARSE_SCORE_WEIGHT, RERANK_SCORE_WEIGHT,
    RRF_K_DEFAULT, TOP_K_DEFAULT, CANDIDATE_LIMIT_DEFAULT,
    RERANK_QUERY_MAX_LENGTH, RERANK_RESUME_WEIGHT_DEFAULT, RERANK_INTENT_WEIGHT_DEFAULT
)

# 日志配置
logger = logging.getLogger(__name__)


# ----------------------------- Pydantic 模型 -----------------------------
class LocationItem(BaseModel):
    """位置项"""
    province: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None


class JobInfoItem(BaseModel):
    """工作信息项（校招格式）"""
    jobTitle: str
    location: Optional[List[LocationItem]] = None


# ----------------------------- 请求转换工具 -----------------------------
def _req_to_filters(req) -> Dict[str, Any]:
    """兼容 Pydantic v1/v2"""
    try:
        return req.model_dump(exclude_none=True)
    except AttributeError:
        try:
            return req.dict(exclude_none=True)
        except Exception as e:
            logger.error(f"Failed to convert request to dict: {e}")
            raise


# ----------------------------- 资源工厂函数 -----------------------------
def create_http_session() -> aiohttp.ClientSession:
    """创建 HTTP Session 工厂"""
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_TOTAL, connect=HTTP_TIMEOUT_CONNECT)
    connector = aiohttp.TCPConnector(limit=HTTP_CONNECTOR_LIMIT, limit_per_host=HTTP_CONNECTOR_LIMIT_PER_HOST)
    return aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
    )


async def create_db_pool(db_config: Dict = None) -> aiomysql.Pool:
    """创建数据库连接池工厂"""
    config = db_config or DB_CONFIG
    return await aiomysql.create_pool(
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


def load_local_embeddings_cache() -> Dict[str, List[float]]:
    """加载本地嵌入缓存"""
    try:
        if os.path.exists(LOCAL_EMBEDDINGS_CACHE_PATH):
            with open(LOCAL_EMBEDDINGS_CACHE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
                data = obj.get("data") if isinstance(obj, dict) else obj
                return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load local embeddings cache: {e}")
    return {}


# ----------------------------- 异步向量化客户端 -----------------------------
class AsyncEmbeddingClient:
    """异步向量化客户端 - 需传入 session 使用"""

    def __init__(self, session: aiohttp.ClientSession, concurrency: int = 5):
        self.session = session
        self.semaphore = asyncio.Semaphore(concurrency)
        self._cache: Dict[str, List[float]] = {}

    async def embed_texts(self, texts: List[str], dimension: int = VECTOR_DIM) -> List[List[float]]:
        """异步批量向量化"""
        if not texts:
            return []

        unique_texts = list(set(t.strip() for t in texts if t and t.strip()))
        if not unique_texts:
            return [[] for _ in texts]

        # 检查内存缓存
        results_map, texts_to_embed = {}, []
        for text in unique_texts:
            if text in self._cache:
                results_map[text] = self._cache[text]
            else:
                texts_to_embed.append(text)

        # 并发调用API
        if texts_to_embed:
            async with self.semaphore:
                embeddings = await asyncio.gather(
                    *(self._embed_single(text, dimension) for text in texts_to_embed),
                    return_exceptions=True,
                )

                for text, emb in zip(texts_to_embed, embeddings):
                    if isinstance(emb, list) and len(emb) == dimension:
                        results_map[text] = emb
                        if any(emb):
                            self._cache[text] = emb

        return [results_map.get(t.strip(), []) for t in texts]

    async def _embed_single(self, text: str, dimension: int) -> List[float]:
        """单个文本向量化"""
        payload = {
            "model": "text-embedding-v4",
            "input": {"texts": [text]},
            "parameters": {"dimension": dimension, "output_type": "dense"}
        }

        for attempt in range(2):
            try:
                async with self.session.post(EMBEDDING_API_URL, json=payload) as resp:
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

        return [0.0] * dimension


# ----------------------------- 异步重排客户端 -----------------------------
class AsyncRerankClient:
    """
    异步重排客户端 - 增强稳定性版本
    
    改进点：
    1. 指数退避重试机制（最多3次）
    2. 响应格式校验和兜底
    3. 超时控制（默认30秒）
    4. 评分标准化（0-100映射到0-1）
    5. 结构化错误信息
    """

    def __init__(self, session: aiohttp.ClientSession, max_retries: int = 3, timeout: float = 30.0):
        self.session = session
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._last_error = None

    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 100,
        min_score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        异步重排 - 带重试和错误兜底
        
        Args:
            query: 查询文本（简历信息）
            documents: 岗位文档列表
            top_k: 返回前k个结果
            min_score_threshold: 最低分数阈值，低于此值的结果会被过滤
            
        Returns:
            List[Dict]: 每个元素包含 index, score, normalized_score
            即使API失败也返回基于粗排分数的兜底结果，确保服务可用性
        """
        if not documents or not query:
            logger.warning("Rerank输入为空，返回空结果")
            return []

        # 构建请求payload
        payload = {
            "model": "qwen3-rerank",
            "input": {"query": query, "documents": documents},
            "parameters": {
                "top_n": min(top_k, len(documents)), 
                "return_documents": False
            }
        }

        # 指数退避重试
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    RERANK_API_URL, 
                    json=payload, 
                    timeout=self.timeout
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = self._parse_response(data, len(documents))
                        if results:
                            logger.info(f"Rerank成功(尝试{attempt+1}/{self.max_retries}): 返回{len(results)}个结果")
                            return self._filter_and_normalize(results, min_score_threshold)
                        else:
                            logger.warning(f"Rerank返回空结果，尝试重试({attempt+1}/{self.max_retries})")
                    else:
                        error_text = await resp.text()
                        logger.warning(f"Rerank API错误 {resp.status}: {error_text[:200]}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Rerank超时(尝试{attempt+1}/{self.max_retries})")
            except Exception as e:
                logger.warning(f"Rerank请求异常(尝试{attempt+1}/{self.max_retries}): {e}")
            
            # 指数退避等待
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.info(f"等待{wait_time}秒后重试...")
                await asyncio.sleep(wait_time)

        # 所有重试失败，返回兜底结果
        logger.error(f"Rerank在{self.max_retries}次尝试后失败，返回兜底结果")
        return self._fallback_results(len(documents))

    def _parse_response(self, data: dict, doc_count: int) -> List[Dict]:
        """解析API响应，处理各种格式"""
        if not isinstance(data, dict):
            logger.warning(f"Rerank响应格式错误: 不是字典")
            return []
        
        # 尝试多种可能的响应路径
        results = None
        
        # 路径1: output.results (标准格式)
        if "output" in data and isinstance(data["output"], dict):
            results = data["output"].get("results")
        
        # 路径2: results (简化格式)
        if results is None and "results" in data:
            results = data["results"]
        
        # 路径3: data.results (某些版本)
        if results is None and "data" in data and isinstance(data["data"], dict):
            results = data["data"].get("results")
        
        if not isinstance(results, (list, tuple)):
            logger.warning(f"Rerank响应中未找到有效results字段")
            return []
        
        parsed = []
        for r in results:
            if not isinstance(r, dict):
                continue
                
            index = r.get("index")
            if index is None:
                continue
                
            # 尝试多种可能的score字段名
            score = (
                r.get("relevance_score") or 
                r.get("score") or 
                r.get("relevance") or
                r.get("similarity") or
                0.0
            )
            
            # 确保score是数字
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            
            parsed.append({
                "index": int(index),
                "score": score,
                "raw": r  # 保留原始数据用于调试
            })
        
        return parsed

    def _filter_and_normalize(
        self, 
        results: List[Dict], 
        min_threshold: float
    ) -> List[Dict]:
        """过滤低分结果并标准化分数"""
        if not results:
            return []
        
        # 获取分数范围用于标准化
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        normalized = []
        for r in results:
            # 标准化到0-1范围
            if score_range > 0:
                norm_score = (r["score"] - min_score) / score_range
            else:
                norm_score = 1.0 if r["score"] > 0 else 0.0
            
            # 应用阈值过滤
            if norm_score >= min_threshold:
                normalized.append({
                    "index": r["index"],
                    "score": r["score"],  # 原始分数
                    "normalized_score": round(norm_score, 4),  # 标准化分数
                })
        
        # 按标准化分数排序
        normalized.sort(key=lambda x: x["normalized_score"], reverse=True)
        return normalized

    def _fallback_results(self, doc_count: int) -> List[Dict]:
        """
        生成兜底结果 - 当API完全失败时使用
        返回均匀分布的分数，确保服务可用性
        """
        logger.warning(f"使用Rerank兜底策略: {doc_count}个文档")
        
        # 生成线性递减的分数（0.8 -> 0.2）
        fallback = []
        for i in range(doc_count):
            # 线性插值：第一个0.8，最后一个0.2
            score = 0.8 - (0.6 * i / max(1, doc_count - 1))
            fallback.append({
                "index": i,
                "score": round(score, 4),
                "normalized_score": round(score, 4),
                "is_fallback": True  # 标记为兜底结果
            })
        
        return fallback

    def get_last_error(self) -> Optional[str]:
        """获取最后一次错误信息，用于外部诊断"""
        return self._last_error



# ----------------------------- 简历向量构建器 -----------------------------
class ResumeVectorBuilder:
    """简历向量构建器 - 多维度加权融合（附件简历格式）"""

    FIELD_WEIGHTS = RESUME_FIELD_WEIGHTS

    @staticmethod
    def extract_skills_text(resume: dict) -> Tuple[str, float]:
        """提取技能文本"""
        weight = ResumeVectorBuilder.FIELD_WEIGHTS["skills"]
        skills_parts = []

        skills = resume.get("skills", [])
        if isinstance(skills, list) and skills:
            skills_parts.extend([str(s).strip() for s in skills if s])
        elif isinstance(skills, str) and skills.strip():
            skills_parts.append(skills.strip())

        professional_skills = resume.get("professional_skills", {})
        if isinstance(professional_skills, dict):
            for category, skill_list in professional_skills.items():
                if isinstance(skill_list, list):
                    skills_parts.extend([f"{category}:{s}" for s in skill_list if s])
                elif isinstance(skill_list, str):
                    skills_parts.append(f"{category}:{skill_list}")

        tech_stack = ResumeVectorBuilder._extract_tech_from_projects(resume)
        if tech_stack:
            skills_parts.extend(tech_stack)
            weight += 0.5

        if not skills_parts:
            return "", 0.0

        unique_skills = list(dict.fromkeys(skills_parts))
        return " ".join(unique_skills[:20]), weight

    @staticmethod
    def _extract_tech_from_projects(resume: dict) -> List[str]:
        """从项目经历中提取技术关键词"""
        tech_keywords = []
        projects = resume.get("project_exp", []) or []

        for proj in projects[:3]:
            if not isinstance(proj, dict):
                continue
            content = proj.get("content", "") or proj.get("description", "") or ""
            tech_patterns = [
                r'Python|Java|C\+\+|Go|Rust|JavaScript|TypeScript',
                r'Spring|Django|Flask|React|Vue|Angular',
                r'MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch',
                r'Docker|Kubernetes|AWS|阿里云|腾讯云',
                r'TensorFlow|PyTorch|机器学习|深度学习',
            ]
            for pattern in tech_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                tech_keywords.extend(matches)

        return list(dict.fromkeys(tech_keywords))

    @staticmethod
    def extract_project_text(resume: dict) -> Tuple[str, float]:
        """提取项目经历文本"""
        weight = ResumeVectorBuilder.FIELD_WEIGHTS["project_exp"]
        projects = resume.get("project_exp", []) or []

        if not projects:
            return "", 0.0

        project_parts = []
        for proj in projects[:2]:
            if not isinstance(proj, dict):
                continue

            name = proj.get("name", "") or proj.get("project_name", "")
            role = proj.get("role", "") or proj.get("position", "")
            content = proj.get("content", "") or proj.get("description", "")

            parts = []
            if name:
                parts.append(f"项目:{name}")
            if role:
                parts.append(f"职责:{role}")
            if content:
                key_sentences = ResumeVectorBuilder._extract_key_sentences(content)
                parts.append(f"内容:{'。'.join(key_sentences[:2])}")

            if parts:
                project_parts.append(" | ".join(parts))

        return "\n".join(project_parts), weight

    @staticmethod
    def _extract_key_sentences(text: str, max_sentences: int = 3) -> List[str]:
        """提取关键句子"""
        if not text:
            return []

        sentences = re.split(r'[。；\n]', text)
        key_sentences = []

        priority_keywords = [
            "负责", "开发", "设计", "实现", "优化", "搭建", "维护",
            "使用", "采用", "基于", "运用",
            "提升", "提高", "降低", "减少", "增加", "实现",
            "独立", "主导", "参与", "带领"
        ]

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            score = sum(1 for kw in priority_keywords if kw in sent)
            if score > 0:
                key_sentences.append((sent, score))

        key_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in key_sentences[:max_sentences]]

    @staticmethod
    def extract_internship_text(resume: dict) -> Tuple[str, float]:
        """提取实习经历文本"""
        weight = ResumeVectorBuilder.FIELD_WEIGHTS["internship_exp"]
        internships = resume.get("internship_exp", []) or resume.get("work_exp", []) or []

        if not internships:
            return "", 0.0

        parts = []
        for intern in internships[:2]:
            if not isinstance(intern, dict):
                continue

            company = intern.get("company", "") or intern.get("company_name", "")
            position = intern.get("position", "") or intern.get("job_title", "")
            content = intern.get("content", "") or intern.get("description", "")

            text_parts = []
            if company:
                text_parts.append(f"公司:{company}")
            if position:
                text_parts.append(f"岗位:{position}")
            if content:
                key_sentences = ResumeVectorBuilder._extract_key_sentences(content, max_sentences=2)
                text_parts.append(f"工作:{'。'.join(key_sentences)}")

            if text_parts:
                parts.append(" | ".join(text_parts))

        return "\n".join(parts), weight

    @staticmethod
    def extract_intent_text(resume: dict) -> Tuple[str, float]:
        """提取求职意向文本"""
        weight = ResumeVectorBuilder.FIELD_WEIGHTS["intent_position"]
        basic = resume.get("basic_info", {}) or {}

        intent_parts = []

        intent_positions = basic.get("intent", []) or basic.get("intent_positions", [])
        if isinstance(intent_positions, list):
            intent_parts.extend([f"期望岗位:{p}" for p in intent_positions[:3]])
        elif isinstance(intent_positions, str) and intent_positions:
            intent_parts.append(f"期望岗位:{intent_positions}")

        intent_industry = basic.get("intent_industry", []) or basic.get("industry", [])
        if isinstance(intent_industry, list):
            intent_parts.extend([f"期望行业:{i}" for i in intent_industry[:2]])
        elif isinstance(intent_industry, str) and intent_industry:
            intent_parts.append(f"期望行业:{intent_industry}")

        return " ".join(intent_parts), weight if intent_parts else 0.0

    @staticmethod
    def extract_education_text(resume: dict) -> Tuple[str, float]:
        """提取教育背景文本"""
        weight_major = ResumeVectorBuilder.FIELD_WEIGHTS["major"]
        weight_courses = ResumeVectorBuilder.FIELD_WEIGHTS["courses"]

        basic = resume.get("basic_info", {}) or {}
        education = resume.get("education", []) or []

        parts = []

        major = basic.get("major", "")
        if major:
            parts.append((f"专业:{major}", weight_major))

        courses = basic.get("courses", [])
        if isinstance(courses, list) and courses:
            course_str = "、".join([str(c) for c in courses[:8]])
            parts.append((f"主修课程:{course_str}", weight_courses))
        elif isinstance(courses, str) and courses:
            parts.append((f"主修课程:{courses}", weight_courses))

        if education and isinstance(education, list):
            edu = education[0]
            if isinstance(edu, dict):
                school = edu.get("school", "")
                degree = edu.get("degree", "")
                if school or degree:
                    parts.append((f"学历:{school} {degree}", 1.0))

        return parts

    @staticmethod
    async def build_resume_vector(resume: dict, embedding_client: AsyncEmbeddingClient, params: Dict = None) -> np.ndarray:
        """构建简历综合向量"""
        if not resume or not isinstance(resume, dict):
            return ResumeVectorBuilder._get_default_vector()

        texts_to_embed = []
        weights = []

        skills_text, skills_weight = ResumeVectorBuilder.extract_skills_text(resume)
        if skills_text:
            texts_to_embed.append(skills_text)
            weights.append(skills_weight)

        proj_text, proj_weight = ResumeVectorBuilder.extract_project_text(resume)
        if proj_text:
            texts_to_embed.append(proj_text)
            weights.append(proj_weight)

        intent_text, intent_weight = ResumeVectorBuilder.extract_intent_text(resume)
        if intent_text:
            texts_to_embed.append(intent_text)
            weights.append(intent_weight)

        edu_parts = ResumeVectorBuilder.extract_education_text(resume)
        for text, weight in edu_parts:
            if text:
                texts_to_embed.append(text)
                weights.append(weight)

        if not texts_to_embed:
            logger.warning("简历无有效内容，返回默认向量")
            return ResumeVectorBuilder._get_default_vector()

        embeddings = await embedding_client.embed_texts(texts_to_embed)

        valid_vectors = []
        valid_weights = []

        for emb, weight in zip(embeddings, weights):
            if emb and len(emb) == EMBEDDING_DIM and any(emb):
                vec = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                valid_vectors.append(vec)
                valid_weights.append(weight)

        if not valid_vectors:
            return ResumeVectorBuilder._get_default_vector()

        V = np.vstack(valid_vectors)
        W = np.asarray(valid_weights, dtype=np.float32).reshape(-1, 1)
        sum_w = np.sum(W)

        if sum_w == 0:
            return ResumeVectorBuilder._get_default_vector()

        weighted_vec = np.sum(V * W, axis=0) / sum_w

        final_norm = np.linalg.norm(weighted_vec)
        if final_norm > 0:
            weighted_vec = weighted_vec / final_norm

        return weighted_vec.astype(np.float32)

    @staticmethod
    def _get_default_vector() -> np.ndarray:
        """获取默认向量"""
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


# ----------------------------- 在线简历向量构建器 -----------------------------
class OnlineResumeVectorBuilder:
    """在线简历向量构建器 - 适配 pr_resume 数据库字段"""

    FIELD_WEIGHTS = ONLINE_RESUME_FIELD_WEIGHTS

    @staticmethod
    def _parse_json_field(field_value) -> list:
        """安全解析 JSON 字段"""
        if not field_value:
            return []
        if isinstance(field_value, list):
            return field_value
        if isinstance(field_value, str):
            try:
                data = json.loads(field_value)
                return data if isinstance(data, list) else []
            except:
                return [{"content": field_value}]
        return []

    @staticmethod
    def extract_describe_text(resume_item: Dict) -> Tuple[str, float]:
        """提取自我介绍 - 这是非常重要的字段，通常包含求职意向和核心优势"""
        describe = resume_item.get("describe", "")
        if not describe or len(describe.strip()) < 5:
            return "", 0.0
        
        # 自我介绍权重很高，因为包含用户的自我定位和核心卖点
        weight = 4.0
        
        # 提取关键句子，去除过长内容
        text = describe.strip()[:500]  # 限制长度避免噪声
        return f"自我介绍: {text}", weight

    @staticmethod
    def extract_skills_text(resume_item: Dict) -> Tuple[str, float]:
        """提取技能文本 - 优化版本，区分证书和技能"""
        weight = OnlineResumeVectorBuilder.FIELD_WEIGHTS["skill"]
        skills_data = OnlineResumeVectorBuilder._parse_json_field(resume_item.get("skill"))
        
        software_skills = []  # 软件/硬技能
        language_skills = []  # 语言技能
        other_skills = []     # 其他技能
        
        # 技能分类关键词
        software_keywords = ["word", "excel", "ppt", "ps", "python", "java", "sql", "office", 
                            "vlookup", "countif", "数据透视表", "ppt", "数据分析"]
        language_keywords = ["普通话", "英语", "cet", "四级", "六级", "雅思", "托福", "日语"]
        
        for s in skills_data:
            name = ""
            if isinstance(s, dict):
                name = s.get("certificateName") or s.get("skillName") or s.get("name", "")
            else:
                name = str(s).strip()
            
            if not name:
                continue
                
            name_lower = name.lower()
            if any(kw in name_lower for kw in software_keywords):
                software_skills.append(name)
            elif any(kw in name for kw in language_keywords):
                language_skills.append(name)
            else:
                other_skills.append(name)
        
        # 合并技能，按优先级排序
        all_skills = software_skills + language_skills + other_skills
        
        # 添加专业特长字段
        prof_skill = resume_item.get("profession_skill") or resume_item.get("professional_skills")
        if prof_skill and isinstance(prof_skill, str):
            all_skills.insert(0, prof_skill.strip())  # 专业特长放最前面

        unique_skills = list(dict.fromkeys([s for s in all_skills if s]))
        text = "技能: " + " ".join(unique_skills[:15]) if unique_skills else ""
        return text.strip(), weight

    @staticmethod
    def extract_certificate_text(resume_item: Dict) -> Tuple[str, float]:
        """提取证书信息"""
        certs = OnlineResumeVectorBuilder._parse_json_field(resume_item.get("certificate"))
        if not certs:
            return "", 0.0
        
        cert_names = []
        for c in certs:
            if isinstance(c, dict):
                name = c.get("certificateName") or c.get("name") or c.get("title")
                if name:
                    cert_names.append(str(name))
            elif isinstance(c, str):
                cert_names.append(c)
        
        if not cert_names:
            return "", 0.0
        
        unique_certs = list(dict.fromkeys(cert_names))[:5]  # 最多取5个证书
        return f"证书: {' '.join(unique_certs)}", 1.5

    @staticmethod
    def extract_practice_text(resume_item: Dict) -> Tuple[str, float]:
        """提取实践经历"""
        practices = OnlineResumeVectorBuilder._parse_json_field(resume_item.get("practice_experience"))
        if not practices:
            return "", 0.0
        
        res = []
        for p in practices[:2]:  # 最多取2条
            if isinstance(p, dict):
                title = p.get("title", "") or p.get("name", "")
                content = p.get("content", "") or p.get("describe", "")
                if title or content:
                    parts = []
                    if title:
                        parts.append(f"实践:{title}")
                    if content:
                        key_sents = ResumeVectorBuilder._extract_key_sentences(content, max_sentences=2)
                        if key_sents:
                            parts.append(f"内容:{'。'.join(key_sents)}")
                    if parts:
                        res.append(" | ".join(parts))
        
        if not res:
            return "", 0.0
        
        return "\n".join(res), 2.0

    @staticmethod
    def extract_project_text(resume_item: Dict) -> Tuple[str, float]:
        """提取项目经历文本"""
        weight = OnlineResumeVectorBuilder.FIELD_WEIGHTS["project_experience"]
        projects = OnlineResumeVectorBuilder._parse_json_field(resume_item.get("project_experience"))
        res = []
        for p in projects[:3]:
            if isinstance(p, dict):
                comp = p.get("companyName", "") or p.get("name", "")
                role = p.get("industry", "") or p.get("role", "")
                job = p.get("jobDescribe", "") or p.get("describe", "") or p.get("content", "")

                parts = []
                if comp:
                    parts.append(f"项目:{comp}")
                if role:
                    parts.append(f"职责:{role}")
                if job:
                    key_sents = ResumeVectorBuilder._extract_key_sentences(job, max_sentences=2)
                    if key_sents:
                        parts.append(f"内容:{'。'.join(key_sents)}")
                    else:
                        parts.append(f"内容:{str(job)[:50]}")

                if parts:
                    res.append(" | ".join(parts))

        return "\n".join(res) if res else "", weight

    @staticmethod
    def extract_work_text(resume_item: Dict) -> Tuple[str, float]:
        """提取实习/工作经历文本"""
        work_exp = resume_item.get("work_experience")
        intern_exp = resume_item.get("internship_experience")

        exp_list = []
        if work_exp:
            exp_list.extend(OnlineResumeVectorBuilder._parse_json_field(work_exp))
        if intern_exp:
            exp_list.extend(OnlineResumeVectorBuilder._parse_json_field(intern_exp))

        weight = OnlineResumeVectorBuilder.FIELD_WEIGHTS["work_experience"] if work_exp else OnlineResumeVectorBuilder.FIELD_WEIGHTS["internship_experience"]

        res = []
        for w in exp_list[:3]:
            if isinstance(w, dict):
                comp = w.get("companyName", "") or w.get("name", "")
                pos = w.get("industry", "") or w.get("position", "")
                job = w.get("jobDescribe", "") or w.get("describe", "") or w.get("content", "")

                parts = []
                if comp:
                    parts.append(f"公司:{comp}")
                if pos:
                    parts.append(f"岗位:{pos}")
                if job:
                    key_sents = ResumeVectorBuilder._extract_key_sentences(job, max_sentences=2)
                    if key_sents:
                        parts.append(f"工作:{'。'.join(key_sents)}")
                    else:
                        parts.append(f"工作:{str(job)[:50]}")

                if parts:
                    res.append(" | ".join(parts))

        return "\n".join(res) if res else "", weight

    @staticmethod
    def extract_education_text(resume_item: Dict) -> Tuple[str, float]:
        """提取教育信息"""
        weight = OnlineResumeVectorBuilder.FIELD_WEIGHTS["education_experience"]
        edus = OnlineResumeVectorBuilder._parse_json_field(resume_item.get("education_experience"))
        res = []
        for e in edus:
            if isinstance(e, dict):
                school = e.get("schoolName", "")
                major = e.get("specialtyName", "")
                edu = e.get("education", "")
                if school or major:
                    res.append(f"教育: {school} {major} {edu}".strip())
        return "\n".join(res) if res else "", weight

    @staticmethod
    def extract_intent_text(resume_item: Dict) -> Tuple[str, float]:
        """提取意向文本 - 融合多个字段"""
        intent_parts = []
        
        # 1. resume_title - 简历名称/求职意向
        title = resume_item.get("resume_title", "")
        if title and len(title) < 30:  # 过滤过长的文件名式标题
            # 过滤掉明显是文件名的
            skip_keywords = ["的简历", "resume", "cv", "我的简历", "个人简历", 
                           ".pdf", ".doc", ".docx", "副本", "复制"]
            if not any(kw in title.lower() for kw in skip_keywords):
                intent_parts.append(f"意向岗位:{title}")
        
        # 2. profession - 职业/专业方向
        profession = resume_item.get("profession", "")
        if profession:
            intent_parts.append(f"职业方向:{profession}")
        
        # 3. industry - 意向行业（逗号分隔）
        industry = resume_item.get("industry", "")
        if industry:
            industries = [i.strip() for i in str(industry).split(",") if i.strip()][:3]
            if industries:
                intent_parts.append(f"意向行业:{'、'.join(industries)}")
        
        # 4. work_city - 期望城市
        work_city = resume_item.get("work_city", "")
        if work_city:
            intent_parts.append(f"期望城市:{work_city}")
        
        if not intent_parts:
            return "", 0.0
        
        # 提升权重，因为求职意向对匹配非常关键
        return "\n".join(intent_parts), 4.0

    @staticmethod
    def extract_major_text(resume_item: Dict) -> Tuple[str, float]:
        """提取专业文本"""
        weight = OnlineResumeVectorBuilder.FIELD_WEIGHTS["profession"]
        major = resume_item.get("profession", "")
        if not major:
            return "", 0.0
        return f"专业: {major}", weight

    @staticmethod
    async def build_vector(resume_item: Dict, embedding_client: AsyncEmbeddingClient) -> np.ndarray:
        """构建在线简历的融合向量 - 增强版"""
        texts_to_embed = []
        weights = []

        # 1. 自我介绍 - 最重要，通常包含求职意向和核心优势
        t_desc, w_desc = OnlineResumeVectorBuilder.extract_describe_text(resume_item)
        if t_desc:
            texts_to_embed.append(t_desc)
            weights.append(w_desc)

        # 2. 求职意向 - 多字段融合
        t_intent, w_intent = OnlineResumeVectorBuilder.extract_intent_text(resume_item)
        if t_intent:
            texts_to_embed.append(t_intent)
            weights.append(w_intent)

        # 3. 技能
        t1, w1 = OnlineResumeVectorBuilder.extract_skills_text(resume_item)
        if t1:
            texts_to_embed.append(t1)
            weights.append(w1)

        # 4. 项目经历
        t2, w2 = OnlineResumeVectorBuilder.extract_project_text(resume_item)
        if t2:
            texts_to_embed.append(t2)
            weights.append(w2)

        # 5. 实习/工作经历
        t3, w3 = OnlineResumeVectorBuilder.extract_work_text(resume_item)
        if t3:
            texts_to_embed.append(t3)
            weights.append(w3)

        # 6. 教育背景
        t_edu, w_edu = OnlineResumeVectorBuilder.extract_education_text(resume_item)
        if t_edu:
            texts_to_embed.append(t_edu)
            weights.append(w_edu)

        # 7. 专业方向
        t5, w5 = OnlineResumeVectorBuilder.extract_major_text(resume_item)
        if t5:
            texts_to_embed.append(t5)
            weights.append(w5)

        # 8. 证书
        t_cert, w_cert = OnlineResumeVectorBuilder.extract_certificate_text(resume_item)
        if t_cert:
            texts_to_embed.append(t_cert)
            weights.append(w_cert)

        # 9. 实践经历
        t_prac, w_prac = OnlineResumeVectorBuilder.extract_practice_text(resume_item)
        if t_prac:
            texts_to_embed.append(t_prac)
            weights.append(w_prac)

        if not texts_to_embed:
            logger.warning("在线简历无有效内容，返回默认向量")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        embeddings = await embedding_client.embed_texts(texts_to_embed)

        valid_vectors = []
        valid_weights = []
        for emb, w in zip(embeddings, weights):
            if emb and len(emb) == EMBEDDING_DIM:
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    valid_vectors.append(vec / norm)
                    valid_weights.append(w)

        if not valid_vectors:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        V = np.vstack(valid_vectors)
        W = np.array(valid_weights, dtype=np.float32).reshape(-1, 1)
        weighted_vec = np.sum(V * W, axis=0) / np.sum(W)

        final_norm = np.linalg.norm(weighted_vec)
        if final_norm > 0:
            weighted_vec = weighted_vec / final_norm

        return weighted_vec.astype(np.float32)

    @staticmethod
    def build_sparse_text(resume_item: Dict) -> str:
        """构建稀疏检索文本 - 增强版"""
        sections = []

        # 自我介绍 - 高权重重复
        t_desc, w_desc = OnlineResumeVectorBuilder.extract_describe_text(resume_item)
        if t_desc:
            sections.extend([t_desc] * int(w_desc))

        # 求职意向 - 最高权重
        t_intent, w_intent = OnlineResumeVectorBuilder.extract_intent_text(resume_item)
        if t_intent:
            sections.extend([t_intent] * int(w_intent))

        # 技能
        t1, w1 = OnlineResumeVectorBuilder.extract_skills_text(resume_item)
        if t1:
            sections.extend([t1] * int(w1))

        # 项目经历
        t2, w2 = OnlineResumeVectorBuilder.extract_project_text(resume_item)
        if t2:
            sections.extend([t2] * int(w2))

        # 实习/工作经历
        t3, w3 = OnlineResumeVectorBuilder.extract_work_text(resume_item)
        if t3:
            sections.extend([t3] * int(w3))

        # 教育背景
        t_edu, w_edu = OnlineResumeVectorBuilder.extract_education_text(resume_item)
        if t_edu:
            sections.extend([t_edu] * int(w_edu))

        # 专业方向
        t5, w5 = OnlineResumeVectorBuilder.extract_major_text(resume_item)
        if t5:
            sections.extend([t5] * int(w5))

        # 证书
        t_cert, w_cert = OnlineResumeVectorBuilder.extract_certificate_text(resume_item)
        if t_cert:
            sections.extend([t_cert] * int(w_cert))

        # 实践经历
        t_prac, w_prac = OnlineResumeVectorBuilder.extract_practice_text(resume_item)
        if t_prac:
            sections.extend([t_prac] * int(w_prac))

        return "\n".join(sections)


def build_resume_text(resume: dict) -> str:
    """构建简历稀疏文本"""
    if not resume or not isinstance(resume, dict):
        return ""

    sections = []

    skills_text, _ = ResumeVectorBuilder.extract_skills_text(resume)
    if skills_text:
        sections.extend([skills_text] * 3)

    intent_text, _ = ResumeVectorBuilder.extract_intent_text(resume)
    if intent_text:
        sections.extend([intent_text] * 2)

    proj_text, _ = ResumeVectorBuilder.extract_project_text(resume)
    if proj_text:
        sections.append(proj_text)

    basic = resume.get("basic_info", {}) or {}
    major = basic.get("major", "")
    if major:
        sections.extend([f"专业:{major}"] * 2)

    courses = basic.get("courses", [])
    if isinstance(courses, list) and courses:
        sections.append("课程:" + "、".join([str(c) for c in courses]))
    elif isinstance(courses, str) and courses:
        sections.append(f"课程:{courses}")

    return "\n".join(sections)



# ----------------------------- 重排查询构建器 -----------------------------
class RerankQueryBuilder:
    """构建高质量重排查询文本"""

    @staticmethod
    def build_query(
            resume: dict = None,
            job_titles: List[str] = None,
            job_info: List[Dict] = None,
            resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
            intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT,
            max_length: int = RERANK_QUERY_MAX_LENGTH
    ) -> str:
        """构建重排查询文本"""
        resume_query = RerankQueryBuilder._build_resume_query(resume)
        intent_query = RerankQueryBuilder._build_intent_query(job_titles, job_info)

        if resume_query and intent_query:
            resume_ratio = resume_weight / (resume_weight + intent_weight)
            intent_ratio = intent_weight / (resume_weight + intent_weight)

            resume_max_len = int(max_length * resume_ratio * 0.8)
            intent_max_len = int(max_length * intent_ratio * 0.8)

            resume_query = RerankQueryBuilder._truncate_text(resume_query, resume_max_len)
            intent_query = RerankQueryBuilder._truncate_text(intent_query, intent_max_len)

            parts = []
            if intent_weight >= resume_weight:
                parts.append(f"【求职意向】(权重{intent_weight:.0%})\n{intent_query}")
                parts.append(f"【简历信息】(权重{resume_weight:.0%})\n{resume_query}")
            else:
                parts.append(f"【简历信息】(权重{resume_weight:.0%})\n{resume_query}")
                parts.append(f"【求职意向】(权重{intent_weight:.0%})\n{intent_query}")

            return "\n\n".join(parts)

        elif resume_query:
            return resume_query
        elif intent_query:
            return intent_query

        return ""

    @staticmethod
    def _build_resume_query(resume: dict) -> str:
        """构建简历部分查询文本"""
        if not resume or not isinstance(resume, dict):
            return ""

        parts = []

        skills_text, _ = ResumeVectorBuilder.extract_skills_text(resume)
        if skills_text:
            skills_short = skills_text[:200] if len(skills_text) > 200 else skills_text
            parts.append(f"技能：{skills_short}")

        proj_text, _ = ResumeVectorBuilder.extract_project_text(resume)
        if proj_text:
            lines = proj_text.split('\n')
            short_proj = []
            for line in lines[:2]:
                line = line.replace("项目:", "").replace("职责:", " ").replace("内容:", " ")
                short_proj.append(line[:100])
            if short_proj:
                parts.append(f"项目：{'；'.join(short_proj)}")

        intern_text, _ = ResumeVectorBuilder.extract_internship_text(resume)
        if intern_text:
            lines = intern_text.split('\n')
            short_intern = []
            for line in lines[:1]:
                line = line.replace("公司:", "").replace("岗位:", " ").replace("工作:", " ")
                short_intern.append(line[:80])
            if short_intern:
                parts.append(f"实习：{'；'.join(short_intern)}")

        intent_text, _ = ResumeVectorBuilder.extract_intent_text(resume)
        if intent_text:
            intent_short = intent_text.replace("期望岗位:", "").replace("期望行业:", " ")
            parts.append(f"意向：{intent_short[:100]}")

        basic = resume.get("basic_info", {}) or {}
        major = basic.get("major", "")
        if major:
            parts.append(f"专业：{major}")

        return "\n".join(parts)

    @staticmethod
    def _build_intent_query(job_titles: List[str] = None, job_info: List[Dict] = None) -> str:
        """构建求职意向部分查询文本"""
        parts = []

        if job_titles and isinstance(job_titles, list):
            unique_titles = list(dict.fromkeys([t for t in job_titles if t and str(t).strip()]))[:5]
            if unique_titles:
                parts.append(f"期望岗位：{'、'.join(unique_titles)}")

        if job_info and isinstance(job_info, list):
            all_titles = []
            all_locations = []

            for item in job_info:
                if isinstance(item, dict):
                    title = item.get("jobTitle", "").strip()
                    if title:
                        all_titles.append(title)

                    locations = item.get("location", [])
                    if locations and isinstance(locations, list):
                        for loc in locations:
                            if isinstance(loc, dict):
                                city = loc.get("city", "")
                                if city:
                                    all_locations.append(city)

            unique_titles = list(dict.fromkeys(all_titles))[:3]
            unique_locations = list(dict.fromkeys(all_locations))[:3]

            if unique_titles:
                parts.append(f"期望岗位：{'、'.join(unique_titles)}")
            if unique_locations:
                parts.append(f"期望城市：{'、'.join(unique_locations)}")

        return "\n".join(parts)

    @staticmethod
    def _truncate_text(text: str, max_length: int) -> str:
        """截断文本"""
        if len(text) <= max_length:
            return text
        truncated = text[:max_length]
        last_punct = max(truncated.rfind('，'), truncated.rfind('。'), truncated.rfind('；'))
        if last_punct > max_length * 0.7:
            truncated = truncated[:last_punct + 1]
        return truncated + "..."


class OnlineRerankQueryBuilder:
    """针对在线简历格式的重排查询构建器 - 增强版"""

    @staticmethod
    def build_query(
            resume: dict = None,
            job_titles: List[str] = None,
            job_info: List[Dict] = None,
            resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
            intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT,
            max_length: int = RERANK_QUERY_MAX_LENGTH
    ) -> str:
        """构建在线简历的 LLM 查询文本"""
        if not resume:
            return ""

        resume_parts = []

        # 1. 自我介绍 - 最重要
        desc_text, _ = OnlineResumeVectorBuilder.extract_describe_text(resume)
        if desc_text:
            resume_parts.append(f"【自我介绍】\n{desc_text}")

        # 2. 求职意向
        intent_text, _ = OnlineResumeVectorBuilder.extract_intent_text(resume)
        if intent_text:
            resume_parts.append(f"【求职意向】\n{intent_text}")

        # 3. 技能
        skill_text, _ = OnlineResumeVectorBuilder.extract_skills_text(resume)
        if skill_text:
            resume_parts.append(f"【技能】\n{skill_text}")

        # 4. 项目经历
        proj_text, _ = OnlineResumeVectorBuilder.extract_project_text(resume)
        if proj_text:
            resume_parts.append(f"【项目经历】\n{proj_text}")

        # 5. 实习/工作经历
        work_text, _ = OnlineResumeVectorBuilder.extract_work_text(resume)
        if work_text:
            resume_parts.append(f"【工作经历】\n{work_text}")

        # 6. 教育背景
        edu_text, _ = OnlineResumeVectorBuilder.extract_education_text(resume)
        if edu_text:
            resume_parts.append(f"【教育】\n{edu_text}")

        # 7. 证书
        cert_text, _ = OnlineResumeVectorBuilder.extract_certificate_text(resume)
        if cert_text:
            resume_parts.append(f"【证书】\n{cert_text}")

        resume_query = "\n\n".join(resume_parts)
        intent_query = RerankQueryBuilder._build_intent_query(job_titles, job_info)

        if resume_query and intent_query:
            resume_max_len = int(max_length * (resume_weight / (resume_weight + intent_weight)) * 0.8)
            intent_max_len = int(max_length * (intent_weight / (resume_weight + intent_weight)) * 0.8)

            resume_query = RerankQueryBuilder._truncate_text(resume_query, resume_max_len)
            intent_query = RerankQueryBuilder._truncate_text(intent_query, intent_max_len)

            return f"【简历信息】(权重{resume_weight:.0%})\n{resume_query}\n\n【求职意向】(权重{intent_weight:.0%})\n{intent_query}"

        return resume_query or intent_query or ""


# ----------------------------- 智能重排器 -----------------------------
class IntelligentReranker:
    """智能重排器 - 多阶段重排"""

    def __init__(self, rerank_client: AsyncRerankClient):
        self.rerank_client = rerank_client

    async def rerank(
            self,
            resume: dict = None,
            candidates: List[Dict] = None,
            job_titles: List[str] = None,
            job_info: List[Dict] = None,
            resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
            intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT,
            top_k: int = 100
    ) -> List[Dict]:
        """两阶段重排"""
        if not candidates:
            return []

        coarse_scored = self._coarse_rank(resume, candidates, job_titles, job_info, resume_weight, intent_weight)
        final_results = await self._fine_rank(resume, coarse_scored, job_titles, job_info, resume_weight, intent_weight, top_k)

        return final_results

    def _coarse_rank(self, resume: dict, candidates: List[Dict], job_titles: List[str] = None,
                     job_info: List[Dict] = None, resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
                     intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT) -> List[Dict]:
        """粗排阶段 - 针对在线简历优化"""
        if not resume:
            return candidates[:150] if candidates else []
        
        # 使用在线简历的字段提取
        resume_skills = set()
        skills_text, _ = OnlineResumeVectorBuilder.extract_skills_text(resume)
        if skills_text:
            clean_skill = skills_text.replace("技能:", "").strip()
            resume_skills = set(re.split(r'[\s\/\,，、;；]+', clean_skill.lower()))
        
        # 专业方向
        resume_major = (resume.get("profession") or "").lower()
        
        # 自我介绍关键词
        describe = (resume.get("describe") or "").lower()
        describe_keywords = set()
        if describe:
            # 提取自我介绍中的关键词（2字以上）
            import jieba
            words = jieba.lcut(describe)
            describe_keywords = {w for w in words if len(w) >= 2}
        
        # 求职意向
        intent_title = (resume.get("resume_title") or "").lower()
        intent_profession = (resume.get("profession") or "").lower()
        intent_industry = (resume.get("industry") or "").lower()
        
        intent_titles = self._extract_intent_titles(job_titles, job_info)
        # 添加用户简历中的意向
        if intent_title:
            intent_titles.append(intent_title)
        if intent_profession:
            intent_titles.append(intent_profession)
        intent_titles = list(dict.fromkeys(intent_titles))  # 去重
        
        intent_keywords = self._extract_intent_keywords(job_titles, job_info)
        # 从industry提取关键词
        if intent_industry:
            industry_keywords = [k.strip() for k in intent_industry.split(",") if k.strip()]
            intent_keywords.extend(industry_keywords)
        intent_keywords = list(dict.fromkeys(intent_keywords))
        
        # 期望城市
        resume_work_city = (resume.get("work_city") or "").strip()
        
        scored = []
        for cand in candidates:
            job_desc = (cand.get("job_describe", "") or "").lower()
            job_name = (cand.get("job_name", "") or "").strip()

            resume_score = 0.0
            resume_features = {}

            # 1. 技能匹配
            if resume_skills and job_desc:
                skill_match_count = 0
                matched_skills = []
                for skill in resume_skills:
                    if len(skill) >= 2 and skill in job_desc:
                        skill_match_count += 1
                        matched_skills.append(skill)

                if resume_skills:
                    skill_score = min(1.0, skill_match_count / max(1, len(resume_skills) * 0.3))
                    resume_score += skill_score * 0.4
                    resume_features["skill_match"] = round(skill_score, 2)
                    if matched_skills:
                        resume_features["matched_skills"] = matched_skills[:5]
            
            # 2. 专业/职业匹配
            if resume_major and job_desc and resume_major in job_desc:
                resume_score += 0.25
                resume_features["profession_match"] = True
            
            # 3. 自我介绍关键词匹配（新增）
            if describe_keywords and job_desc:
                match_count = sum(1 for kw in describe_keywords if kw in job_desc)
                if describe_keywords:
                    desc_score = min(0.3, match_count / max(1, len(describe_keywords) * 0.1))
                    resume_score += desc_score
                    resume_features["describe_match"] = round(desc_score, 2)

            # 4. RRF分数
            rrf_score = cand.get("rrf_score", 0)
            resume_score += min(rrf_score * 0.3, 0.3)

            intent_score = 0.0
            intent_features = {}

            # 5. 岗位名称匹配（意向）
            if intent_titles:
                title_match_score = self._calc_title_match(job_name, intent_titles)
                intent_score += title_match_score * 0.5
                intent_features["intent_title_match"] = round(title_match_score, 2)

            # 6. 关键词匹配
            if intent_keywords and job_desc:
                skill_match_score = self._calc_skill_match(job_desc, "", intent_keywords)
                intent_score += skill_match_score * 0.3
                intent_features["intent_skill_match"] = round(skill_match_score, 2)

            # 7. 地点匹配 - 使用简历中的 work_city
            if resume_work_city:
                job_city = (cand.get("city", "") or "").strip()
                if job_city == resume_work_city:
                    intent_score += 0.2
                    intent_features["location_match"] = True
            # 同时检查 job_info 中的地点
            intent_locations = self._extract_intent_locations(job_info)
            if intent_locations:
                job_city = (cand.get("city", "") or "").strip()
                if job_city in intent_locations:
                    intent_score += 0.15
                    intent_features["location_match_job_info"] = True

            total_weight = resume_weight + intent_weight
            if total_weight == 0:
                total_weight = 1.0

            normalized_resume_weight = resume_weight / total_weight
            normalized_intent_weight = intent_weight / total_weight

            final_coarse_score = (
                resume_score * normalized_resume_weight +
                intent_score * normalized_intent_weight
            )

            cand["coarse_score"] = round(final_coarse_score, 4)
            cand["resume_score"] = round(resume_score, 4)
            cand["intent_score"] = round(intent_score, 4)
            cand["resume_features"] = resume_features
            cand["intent_features"] = intent_features
            scored.append(cand)

        scored.sort(key=lambda x: x["coarse_score"], reverse=True)

        filtered = [c for c in scored if c["coarse_score"] >= THRESHOLD.COARSE_MIN_SCORE]
        if not filtered:
            filtered = scored[:50]

        return filtered[:150]

    @staticmethod
    def _extract_intent_titles(job_titles: List[str] = None, job_info: List[Dict] = None) -> List[str]:
        """提取意向岗位名称"""
        titles = []
        if job_titles and isinstance(job_titles, list):
            titles.extend([str(t).strip().lower() for t in job_titles if t and str(t).strip()])
        if job_info and isinstance(job_info, list):
            for item in job_info:
                if isinstance(item, dict):
                    title = item.get("jobTitle", "").strip()
                    if title:
                        titles.append(title.lower())
        return list(dict.fromkeys(titles))

    @staticmethod
    def _extract_intent_keywords(job_titles: List[str] = None, job_info: List[Dict] = None) -> List[str]:
        """提取意向关键词"""
        titles = IntelligentReranker._extract_intent_titles(job_titles, job_info)
        keywords = []
        for title in titles:
            parts = re.split(r'[\s\/\,，、;；]+', title)
            for part in parts:
                part = part.strip()
                if len(part) >= 2:
                    keywords.append(part.lower())
        return list(dict.fromkeys(keywords))

    @staticmethod
    def _extract_intent_locations(job_info: List[Dict] = None) -> List[str]:
        """提取意向地点"""
        locations = []
        if job_info and isinstance(job_info, list):
            for item in job_info:
                if isinstance(item, dict):
                    locs = item.get("location", [])
                    if locs and isinstance(locs, list):
                        for loc in locs:
                            if isinstance(loc, dict):
                                city = loc.get("city", "").strip()
                                if city:
                                    locations.append(city)
        return list(dict.fromkeys(locations))

    @staticmethod
    def _calc_title_match(job_name: str, intent_titles: List[str]) -> float:
        """计算岗位名称匹配度"""
        if not job_name or not intent_titles:
            return 0.0

        job_name_lower = job_name.lower()
        for intent_title in intent_titles:
            intent_lower = intent_title.lower()
            if job_name_lower == intent_lower:
                return 1.0
            if job_name_lower in intent_lower or intent_lower in job_name_lower:
                return 0.6
            job_keywords = set(re.split(r'[\s\/\,，、;；]+', job_name_lower))
            intent_keywords = set(re.split(r'[\s\/\,，、;；]+', intent_lower))
            if job_keywords and intent_keywords:
                common = job_keywords & intent_keywords
                union = job_keywords | intent_keywords
                if union:
                    jaccard = len(common) / len(union)
                    if jaccard >= 0.5:
                        return 0.4
        return 0.0

    @staticmethod
    def _calc_skill_match(job_desc: str, job_req: str, intent_keywords: List[str]) -> float:
        """计算技能匹配度"""
        if not intent_keywords or (not job_desc and not job_req):
            return 0.0

        combined_text = f"{job_desc} {job_req}".lower()
        match_count = 0
        for keyword in intent_keywords:
            if len(keyword) >= 2 and keyword in combined_text:
                match_count += 1

        if intent_keywords:
            match_ratio = match_count / len(intent_keywords)
            return min(1.0, match_ratio * 1.5)
        return 0.0

    async def _fine_rank(self, resume: dict, candidates: List[Dict], job_titles: List[str] = None,
                         job_info: List[Dict] = None, resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
                         intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT, top_k: int = 100) -> List[Dict]:
        """精排阶段"""
        if not candidates:
            return []

        query = RerankQueryBuilder.build_query(
            resume=resume, job_titles=job_titles, job_info=job_info,
            resume_weight=resume_weight, intent_weight=intent_weight
        )

        if not query:
            return candidates[:top_k]

        documents = []
        for cand in candidates:
            job_desc = cand.get('job_describe', '') or ''
            desc_part = job_desc[:500] if len(job_desc) > 500 else job_desc
            doc_parts = [
                f"岗位：{cand.get('job_name', '')}",
                f"描述与要求：{desc_part}"
            ]
            documents.append("\n".join(doc_parts))

        rerank_results = await self.rerank_client.rerank(query, documents, top_k=len(candidates))

        if not rerank_results:
            return candidates[:top_k]

        rerank_map = {r["index"]: r["score"] for r in rerank_results}

        final_scored = []
        for i, cand in enumerate(candidates):
            rerank_score = rerank_map.get(i, 0)
            coarse_score = cand.get("coarse_score", 0)
            final_score = coarse_score * COARSE_SCORE_WEIGHT + rerank_score * RERANK_SCORE_WEIGHT

            cand["final_score"] = round(final_score, 4)
            cand["rerank_score"] = round(rerank_score, 4)
            final_scored.append(cand)

        filtered = [c for c in final_scored if c["final_score"] >= THRESHOLD.RERANK_MIN_SCORE]
        if not filtered:
            filtered = final_scored[:min(20, len(final_scored))]

        filtered.sort(key=lambda x: x["final_score"], reverse=True)
        return filtered[:top_k]


class OnlineIntelligentReranker:
    """在线简历专用智能重排器"""

    def __init__(self, rerank_client: AsyncRerankClient):
        self.rerank_client = rerank_client

    async def rerank(
            self,
            resume: dict = None,
            candidates: List[Dict] = None,
            job_titles: List[str] = None,
            job_info: List[Dict] = None,
            resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
            intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT,
            top_k: int = 100
    ) -> List[Dict]:
        """两阶段重排"""
        if not candidates:
            return []

        coarse_scored = self._coarse_rank(resume, candidates, job_titles, job_info, resume_weight, intent_weight)
        final_results = await self._fine_rank(resume, coarse_scored, job_titles, job_info, resume_weight, intent_weight, top_k)

        return final_results

    def _coarse_rank(self, resume: dict, candidates: List[Dict], job_titles: List[str] = None,
                     job_info: List[Dict] = None, resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
                     intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT) -> List[Dict]:
        """粗排阶段"""
        resume_skills = set()
        skill_text, _ = OnlineResumeVectorBuilder.extract_skills_text(resume)
        resume_major = (resume.get("profession") or "").lower() if resume else ""

        if skill_text:
            clean_skill = skill_text.replace("技能: ", "")
            resume_skills = set(re.split(r'[\s\/\,，、;；]+', clean_skill.lower()))

        intent_titles = IntelligentReranker._extract_intent_titles(job_titles, job_info)
        intent_keywords = IntelligentReranker._extract_intent_keywords(job_titles, job_info)

        scored = []
        for cand in candidates:
            job_desc = (cand.get("job_describe", "") or "").lower()

            resume_score = 0.0
            resume_features = {}
            if resume_skills and job_desc:
                match_count = sum(1 for s in resume_skills if len(s) >= 2 and s in job_desc)
                s_score = min(1.0, match_count / max(1, len(resume_skills) * 0.3))
                resume_score += s_score * 0.4
                resume_features["skill_match"] = round(s_score, 2)

            if resume_major and job_desc and resume_major in job_desc:
                resume_score += 0.2
                resume_features["major_match"] = True

            resume_score += min(cand.get("rrf_score", 0) * 0.3, 0.3)

            intent_score = 0.0
            intent_features = {}
            job_name = (cand.get("job_name", "") or "").strip()
            if intent_titles:
                t_match = IntelligentReranker._calc_title_match(job_name, intent_titles)
                intent_score += t_match * 0.5
                intent_features["intent_title_match"] = round(t_match, 2)

            if intent_keywords and job_desc:
                k_match = IntelligentReranker._calc_skill_match(job_desc, "", intent_keywords)
                intent_score += k_match * 0.3
                intent_features["intent_skill_match"] = round(k_match, 2)

            total_w = resume_weight + intent_weight or 1.0
            final_c_score = (resume_score * (resume_weight / total_w) + intent_score * (intent_weight / total_w))

            cand.update({
                "coarse_score": round(final_c_score, 4),
                "resume_score": round(resume_score, 4),
                "intent_score": round(intent_score, 4),
                "resume_features": resume_features,
                "intent_features": intent_features
            })
            scored.append(cand)

        scored.sort(key=lambda x: x["coarse_score"], reverse=True)
        filtered = [c for c in scored if c["coarse_score"] >= THRESHOLD.COARSE_MIN_SCORE] or scored[:50]
        return filtered[:150]

    async def _fine_rank(self, resume: dict, candidates: List[Dict], job_titles: List[str] = None,
                         job_info: List[Dict] = None, resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
                         intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT, top_k: int = 100) -> List[Dict]:
        """精排阶段"""
        query = OnlineRerankQueryBuilder.build_query(
            resume=resume, job_titles=job_titles, job_info=job_info,
            resume_weight=resume_weight, intent_weight=intent_weight
        )
        if not query:
            return candidates[:top_k]

        documents = [f"岗位：{c.get('job_name', '')}\n描述与要求：{str(c.get('job_describe', ''))[:500]}" for c in candidates]
        rerank_results = await self.rerank_client.rerank(query, documents, top_k=len(candidates))

        if not rerank_results:
            return candidates[:top_k]

        rerank_map = {r["index"]: r["score"] for r in rerank_results}

        for i, cand in enumerate(candidates):
            r_score = rerank_map.get(i, 0)
            c_score = cand.get("coarse_score", 0)
            cand["final_score"] = round(c_score * COARSE_SCORE_WEIGHT + r_score * RERANK_SCORE_WEIGHT, 4)
            cand["rerank_score"] = round(r_score, 4)

        final_scored = [c for c in candidates if c.get("final_score", 0) >= THRESHOLD.RERANK_MIN_SCORE] or candidates[:20]
        final_scored.sort(key=lambda x: x["final_score"], reverse=True)
        return final_scored[:top_k]



# ----------------------------- 过滤条件构建函数 -----------------------------
def _sanitize_str(value: Optional[str]) -> Optional[str]:
    """清理字符串"""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    return None


def _and_expr(exprs) -> Optional[str]:
    """构建 AND 表达式"""
    exprs = [e for e in exprs if e]
    return " and ".join(f"({e})" for e in exprs) if exprs else None


def _build_locations_filter_milvus(locations: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """构建位置过滤条件"""
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


def _build_refresh_time_filter_milvus(refresh_time: Optional[int]) -> Optional[str]:
    """构建刷新时间过滤条件"""
    if refresh_time is None or not isinstance(refresh_time, int) or refresh_time == 0:
        return None
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    now = int(today_start.timestamp())
    mapping = {1: now - 86400, 2: now - 259200, 3: now - 604800, 4: now - 1209600, 5: now - 2592000}
    gte_ts = mapping.get(refresh_time)
    return f"updated_time >= {gte_ts}" if gte_ts else None


def _build_gender_filter_milvus(gender: Optional[int]) -> Optional[str]:
    """构建性别过滤条件"""
    if gender in (1, 2):
        return f"gender == {gender}"
    return None


def _build_salary_settle_filter_milvus(salary_settle: Optional[str]) -> Optional[str]:
    """构建薪资结算方式过滤条件"""
    s = _sanitize_str(salary_settle)
    if not s or s.upper() == "ANY":
        return None
    return f'salarySettle == "{s.upper()}"' if s.upper() in {"ORDER", "DAY", "WEEK", "MONTH", "OTHER"} else None


def _build_job_opening_state_filter_milvus(state: Optional[int]) -> Optional[str]:
    """构建职位状态过滤条件"""
    if state == 1:
        return "is_open == true"
    if state in (-1, 0):
        return "is_open == false"
    return None


def build_education_filter_milvus(education: Optional[int]) -> Optional[str]:
    """构建学历过滤条件"""
    if education in (1, 2, 3, 4, 5):
        return f"education <= {education}"
    return None


# ----------------------------- 数学工具函数 -----------------------------
def cosine_sim(a, b) -> float:
    """计算余弦相似度"""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _dedupe_keep_order(items):
    """去重并保持顺序"""
    return list(dict.fromkeys(items))


def rrf_fusion(rank_maps: dict, k: int = RRF_K_DEFAULT):
    """RRF 融合"""
    scores = defaultdict(float)
    for results in rank_maps.values():
        for rank, job_id in enumerate(results):
            scores[job_id] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [job_id for job_id, _ in ranked], [score for _, score in ranked]


# ----------------------------- Milvus 查询函数 -----------------------------
async def hybrid_recall_with_rrf_async(
        collection,
        resume_vec: np.ndarray,
        query_text: str,
        embedding_client: AsyncEmbeddingClient,
        filters: Optional[str] = None,
        top_k: int = TOP_K_DEFAULT,
        candidate_limit: int = CANDIDATE_LIMIT_DEFAULT,
        rrf_k: int = RRF_K_DEFAULT
) -> List[Dict[str, Any]]:
    """异步混合召回 - 需传入 collection 和 embedding_client"""
    loop = asyncio.get_running_loop()

    def _sync_search():
        search_requests = []

        vector_request = AnnSearchRequest(
            data=[resume_vec.tolist()],
            anns_field="job_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 128}},
            limit=candidate_limit,
            expr=filters
        )
        search_requests.append(vector_request)

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

        rrf_ranker = RRFRanker(rrf_k)
        search_results = collection.hybrid_search(
            reqs=search_requests,
            rerank=rrf_ranker,
            limit=candidate_limit,
            output_fields=["job_id", "job_name", "job_vector", "job_describe", "province", "city"]
        )

        results_map = {}
        for hits in search_results:
            for hit in hits:
                job_id = hit.entity.get("job_id")
                if job_id is None:
                    continue
                job_id = int(job_id)
                job_vector = np.array(hit.entity.get("job_vector"))
                sim = float(cosine_sim(resume_vec, job_vector)) if job_vector.size > 0 else 0.0

                if sim >= THRESHOLD.RECALL_MIN_SIM:
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

    return await loop.run_in_executor(None, _sync_search)


async def fetch_jobs_by_id_async(collection, job_ids: List[int]) -> List[Dict]:
    """异步批量查询岗位详情 - 需传入 collection"""
    if not job_ids:
        return []

    loop = asyncio.get_running_loop()

    def _sync_query():
        def _format_id(x):
            if isinstance(x, (int,)):
                return str(int(x))
            try:
                if isinstance(x, str) and x.isdigit():
                    return str(int(x))
            except Exception:
                pass
            return '"{}"'.format(str(x))

        formatted = [_format_id(x) for x in job_ids]
        expr = f"job_id in [{', '.join(formatted)}]"
        return collection.query(
            expr=expr,
            output_fields=["job_id", "job_name", "job_describe", "province", "city"]
        )

    return await loop.run_in_executor(None, _sync_query)


# ----------------------------- 数据库查询函数 -----------------------------
async def get_resume_list_by_user_attachment(
        db_pool: aiomysql.Pool,
        user_id: int,
        resume_type: int = 1
) -> List[Dict]:
    """根据用户ID获取附件简历列表 - 需传入 db_pool"""
    try:
        logger.info(f"查询用户简历列表: user_id={user_id}, resume_type={resume_type}")

        async with db_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                if resume_type:
                    sql = """
                    SELECT id, user_id, name, resume_type, work_city, 
                           preview_address, create_time, update_time, post_address
                    FROM pr_resume_parse 
                    WHERE user_id = %s AND resume_type = %s AND is_deleted = 0
                    ORDER BY update_time DESC
                    """
                    await cur.execute(sql, (user_id, resume_type))
                else:
                    sql = """
                    SELECT id, user_id, name, resume_type, work_city, 
                           preview_address, create_time, update_time, post_address
                    FROM pr_resume_parse 
                    WHERE user_id = %s AND is_deleted = 0
                    ORDER BY update_time DESC
                    """
                    await cur.execute(sql, (user_id,))

                rows = await cur.fetchall()

        resume_list = []
        for row in rows:
            resume_item = {
                "id": row["id"],
                "user_id": row["user_id"],
                "name": row["name"] or "",
                "resume_type": row["resume_type"],
                "work_city": row["work_city"],
                "preview_address": row["preview_address"],
                "post_address": row["post_address"]
            }
            resume_list.append(resume_item)

        logger.info(f"查询完成: user_id={user_id}, 共{len(resume_list)}条记录")
        return resume_list

    except Exception as e:
        logger.error(f"查询简历列表失败: user_id={user_id}, error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


async def get_resume_list_by_user_online(
        db_pool: aiomysql.Pool,
        user_id: int,
        work_type: int = 3
) -> List[Dict]:
    """根据用户ID获取在线简历列表 - 需传入 db_pool"""
    try:
        logger.info(f"查询用户在线简历列表: user_id={user_id}, work_type={work_type}")

        async with db_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                sql = """
                SELECT id, user_id, name, gender, photo, id_number, birthday, phone, email, city, 
                       `describe`, industry, profession, work_type, work_city, salary_min, salary_max, 
                       job_status, work_explain, work_start_year, project_experience, internship_experience, 
                       education_experience, practice_experience, certificate, skill, education, work_years, 
                       attach, complete, qq, preview_address, is_deleted, create_time, update_time, 
                       work_experience, resume_title, resume_thumbnail, resume_type, resume_icon, 
                       academic_experience, society_experience, post_address
                FROM pr_resume
                WHERE user_id = %s AND work_type = %s AND is_deleted = 0 AND resume_type = 1
                ORDER BY update_time DESC
                """
                await cur.execute(sql, (user_id, work_type))

                rows = await cur.fetchall()

        resume_list = []
        for row in rows:
            resume_item = {key: row[key] for key in row.keys()}
            resume_list.append(resume_item)

        logger.info(f"查询完成: user_id={user_id}, 共{len(resume_list)}条记录")
        return resume_list

    except Exception as e:
        logger.error(f"查询在线简历列表失败: user_id={user_id}, error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
