# -*- coding: utf-8 -*-
"""
简历岗位投递匹配分数接口
说明：
1. 直接读取 pr_resume_parse 表中已解析的附件简历数据（post_address + work_city），不再重复解析文件。
2. 粗排逻辑针对【附件简历 JSON 结构】和【HR 硬条件】专门优化，完全对齐推荐系统语义，
   同时引入领域匹配、经历文本匹配、城市匹配等投递场景核心要素。
3. 精排阶段复用 common.RerankQueryBuilder + AsyncRerankClient，确保与推荐系统协同一致。
4. 岗位不通过 id 查询，直接传入 job_name / job_describe / job_cities。
5. 返回精简，仅返回最终匹配得分。
"""
import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
import aiomysql
import aiohttp
import numpy as np

# 尝试导入 jieba 分词（可选依赖）
try:
    import jieba
    jieba.setLogLevel(20)  # 关闭日志输出
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    jieba = None

from config import (
    RESUME_TYPE_CAMPUS,
    RESUME_TYPE_PART_TIME,
    COARSE_SCORE_WEIGHT,
    RERANK_SCORE_WEIGHT,
    THRESHOLD,
    RERANK_RESUME_WEIGHT_DEFAULT,
    RERANK_INTENT_WEIGHT_DEFAULT,
    JOB_DETAIL_API_BASE,
    JOB_DETAIL_API_TOKEN,
)
from common import (
    logger,
    create_http_session,
    create_db_pool,
    AsyncEmbeddingClient,
    AsyncRerankClient,
    RerankQueryBuilder,
    ResumeVectorBuilder,
    cosine_sim,
)

# ----------------------------- 常量配置 -----------------------------
# 投递场景下 coarse/rerank 融合权重
# 投递场景权重调整：粗排50%+精排50%
# 原因：
# 1. 投递场景是1对1匹配，粗排特征可解释、稳定
# 2. LLM精排对单candidate打分容易偏保守，集中在40-70分
# 3. 提高粗排权重可以增加区分度，让高匹配真正高分、低匹配真正低分
DELIVERY_COARSE_WEIGHT = 0.50
DELIVERY_RERANK_WEIGHT = 0.50

# 院校关键词库
TOP_SCHOOL_KEYWORDS = {
    "清华", "北大", "复旦", "上海交通", "浙江", "南京", "中国科学技术", "人民",
    "北京航空", "北京理工", "华中科技", "武汉", "中山", "哈尔滨工业", "西安交通",
    "南开", "同济", "东南", "北京师范", "四川", "厦门", "山东", "天津", "华南理工",
    "中南", "电子科技", "湖南", "西北工业", "华东师范", "大连理工", "中国农业",
    "重庆", "兰州", "东北", "吉林", "中国海洋", "西北农林科技", "中央民族",
    "上海财经", "对外经济贸易", "北京邮电", "西安电子", "北京交通", "南京航空",
    "北京科技", "北京化工", "河海", "华北电力", "西南交通", "华东理工", "南京理工",
    "苏州", "华中农业", "武汉理工", "中国传媒", "中国石油", "中国地质", "中国矿业",
    "暨南", "哈尔滨工程", "江南", "南京农业", "东北师范", "陕西师范", "华南师范",
    "西南财经", "中南财经政法", "东华", "北京工业", "北京林业", "北京中医药",
    "首都师范", "北京外国语", "上海外国语", "天津医科", "河北工业", "太原理工",
    "辽宁", "大连海事", "东北林业", "合肥工业", "福州", "南昌", "郑州", "华中师范",
    "湖南师范", "华南农业", "广西", "四川农业", "西南", "贵州", "云南", "西北",
    "长安", "新疆", "海南", "宁夏", "青海", "西藏", "石河子", "南方科技",
}

# Dummy candidates 用于给 rerank 提供对比参照（解决单 candidate 打分偏低问题）
DUMMY_CANDIDATES = [
    {
        "job_id": -1,
        "job_name": "餐饮服务员",
        "job_describe": "负责餐厅顾客接待、点餐、上菜及桌面清洁，保持用餐环境整洁。",
        "city": "",
    },
    {
        "job_id": -2,
        "job_name": "电话销售",
        "job_describe": "通过电话开发新客户，完成销售指标，维护客户关系。",
        "city": "",
    },
]

# ── SKILL_SYNONYMS 和 DOMAIN_KEYWORDS 已移除 ──
# 技能匹配和领域匹配均已改为向量语义匹配（text-embedding-v4），
# 不再依赖硬编码字典，天然覆盖各行各业。
# 旧字典删除标志（保留注释用于追溯，勿恢复）：
# SKILL_SYNONYMS: 技能同义词字典 → 已废弃
# DOMAIN_KEYWORDS: 领域关键词字典 → 已废弃

# 以下保留一个空壳防止旧代码残留引用时报 NameError（不会实际使用）
SKILL_SYNONYMS = {}
DOMAIN_KEYWORDS = {}




# ----------------------------- 服务状态管理 -----------------------------
class ServiceState:
    def __init__(self):
        self.http_session = None
        self.db_pool = None
        self.embedding_client = None
        self.rerank_client = None
        self.delivery_reranker = None

    async def initialize(self):
        logger.info("=== 简历投递匹配分数服务启动，初始化资源 ===")
        self.http_session = create_http_session()
        self.db_pool = await create_db_pool()
        self.embedding_client = AsyncEmbeddingClient(self.http_session)
        self.rerank_client = AsyncRerankClient(self.http_session)
        self.delivery_reranker = AttachmentDeliveryReranker(self.rerank_client)
        logger.info("=== 简历投递匹配分数服务初始化完成 ===")

    async def cleanup(self):
        logger.info("=== 简历投递匹配分数服务关闭，清理资源 ===")
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        if self.db_pool:
            self.db_pool.close()
            await self.db_pool.wait_closed()
        logger.info("=== 简历投递匹配分数服务资源清理完成 ===")


_state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _state.initialize()
    yield
    await _state.cleanup()


app = FastAPI(lifespan=lifespan)


# ----------------------------- Pydantic 模型 -----------------------------
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


# ----------------------------- 数据库查询 -----------------------------
async def get_attachment_resume_by_id(
        db_pool: aiomysql.Pool,
        resume_id: int,
        resume_type: int
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


# ----------------------------- 岗位详情获取 -----------------------------
async def fetch_job_detail(job_id: int, api_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """通过岗位详情接口获取岗位信息"""
    url = f"{JOB_DETAIL_API_BASE}/{job_id}"
    token = api_token if api_token else JOB_DETAIL_API_TOKEN
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with _state.http_session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("code") == 0 and data.get("success"):
                    return data.get("data")
                else:
                    logger.warning(
                        f"岗位详情接口返回失败: job_id={job_id}, "
                        f"code={data.get('code')}, msg={data.get('message') or data.get('msg')}"
                    )
            else:
                body = await resp.text()
                logger.warning(f"获取岗位详情失败: job_id={job_id}, status={resp.status}, body={body[:200]}")
    except Exception as e:
        logger.error(f"获取岗位详情异常: job_id={job_id}, error={str(e)}")
    return None


def parse_job_detail(data: Dict[str, Any]) -> Dict[str, str]:
    """解析岗位详情响应，提取用于匹配的字段"""
    job_name = data.get("jobName", "")
    describe = data.get("describe", "")
    requirement = data.get("requirement", "")
    city = data.get("city", "")

    # 将福利标签和岗位标签合并到描述中，增强匹配信息
    welfare_names = [w.get("welfareName", "") for w in data.get("welfareList", []) if w.get("welfareName")]
    tag_names = [t.get("name", "") for t in data.get("tags", []) if t.get("name")]

    parts = [f"岗位：{job_name}", describe, requirement]
    if welfare_names:
        parts.append("福利标签：" + "、".join(welfare_names))
    if tag_names:
        parts.append("岗位标签：" + "、".join(tag_names))

    job_describe = "\n".join([p.strip() for p in parts if p.strip()])
    return {
        "job_name": job_name,
        "job_describe": job_describe,
        "city": city,
    }


# ----------------------------- HR 硬条件辅助函数 -----------------------------
def _parse_degree_level(degree_str: str) -> int:
    if not degree_str:
        return 0
    d = degree_str.lower()
    if any(k in d for k in ["博士", "phd", "ph.d", "doctor"]):
        return 4
    if any(k in d for k in ["硕士", "研究生", "master", "msc", "mba"]):
        return 3
    if any(k in d for k in ["本科", "学士", "bachelor", "统招", "全日制本科"]):
        return 2
    if any(k in d for k in ["大专", "专科", "高职", "associate", "技校"]):
        return 1
    return 0


def _extract_degree_requirement(job_desc: str) -> int:
    if not job_desc:
        return 0
    d = job_desc.lower()
    if any(k in d for k in ["博士", "phd", "ph.d", "博士学位", "博士研究生"]):
        return 4
    if any(k in d for k in ["硕士", "研究生", "硕士及以上", "研究生在读", "硕士学历", "硕士研究生"]):
        return 3
    if any(k in d for k in ["本科", "学士", "本科及以上", "本科学历", "大学本科", "统招本科", "全日制本科"]):
        return 2
    if any(k in d for k in ["大专", "专科", "大专及以上", "高职", "专科学历"]):
        return 1
    return 0


def _is_top_school(resume: Dict[str, Any]) -> bool:
    basic = resume.get("basic_info", {}) or {}
    schools = []
    if basic.get("school"):
        schools.append(basic["school"])
    for edu in (resume.get("edu_bg") or []):
        if isinstance(edu, dict) and edu.get("school"):
            schools.append(edu["school"])
    for school in schools:
        school_str = school.strip()
        if any(tag in school_str for tag in ["985", "211", "双一流", "C9"]):
            return True
        for keyword in TOP_SCHOOL_KEYWORDS:
            if keyword in school_str:
                return True
    return False


def _filter_soft_skills(text: str) -> str:
    """过滤软性素质噪声词，保留技术/专业相关内容"""
    if not text:
        return ""
    # 移除软性素质词
    filtered = text
    for phrase in SOFT_SKILL_PHRASES:
        filtered = filtered.replace(phrase, "")
    # 清理多余空格
    filtered = re.sub(r'\s+', ' ', filtered).strip()
    return filtered


def _extract_project_text(resume: Dict[str, Any]) -> str:
    """
    提取项目经历原始文本，用于语义向量化
    替代原硬编码tech_patterns的正则匹配
    """
    texts = []
    for proj in (resume.get("project_exp") or [])[:3]:
        if isinstance(proj, dict):
            # 按优先级提取有效字段
            for field in ["responsibility", "content", "title", "tech_stack", "description"]:
                val = proj.get(field, "")
                if val and isinstance(val, str):
                    # 去除日期噪声，保留核心描述
                    cleaned = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', '', val)
                    cleaned = re.sub(r'\d{4}年\d{2}月', '', cleaned)
                    # 过滤软性素质词
                    cleaned = _filter_soft_skills(cleaned)
                    if len(cleaned.strip()) >= 15:
                        texts.append(cleaned.strip())
                        break  # 每个项目取第一个有效字段
    return " | ".join(texts)[:400]


def _extract_work_text(resume: Dict[str, Any]) -> str:
    """
    提取工作经历文本，用于语义向量化
    """
    texts = []
    for work in (resume.get("work_exp") or [])[:2]:
        if isinstance(work, dict):
            for field in ["responsibility", "content", "title", "work_content"]:
                val = work.get(field, "")
                if val and isinstance(val, str):
                    cleaned = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', '', val)
                    # 过滤软性素质词
                    cleaned = _filter_soft_skills(cleaned)
                    if len(cleaned.strip()) >= 15:
                        texts.append(cleaned.strip())
                        break
    return " | ".join(texts)[:300]


def _extract_resume_skills(resume: Dict[str, Any]) -> Dict[str, float]:
    """
    提取简历技能，返回带权重的字典。
    
    权重规则：
    - 1.0: 核心技能（项目经历中高频出现、明确标注"精通"、工作/实习中主要使用的技能）
    - 0.7: 熟练技能（项目经历中出现、标注"熟练"、课程中重点学习的技能）
    - 0.4: 了解技能（仅出现在技能列表、标注"了解/入门"、课程中提及的技能）
    
    Returns:
        Dict[str, float]: 技能名 -> 权重
    """
    skills_weight: Dict[str, float] = {}
    
    # 熟练度关键词映射
    PROFICIENCY_PATTERNS = {
        1.0: [r'精通', r'熟练掌握', r'擅长', r'主要使用', r'核心技能', r'主力'],
        0.7: [r'熟练', r'熟悉', r'掌握', r'常用', r'经验丰富'],
        0.4: [r'了解', r'入门', r'接触过', r'学过', r'会', r'使用过'],
    }
    
    def extract_skill_and_weight(text: str) -> Tuple[str, float]:
        """从技能文本中提取技能名和熟练度权重"""
        text_lower = text.lower()
        weight = 0.7  # 默认熟练
        
        for w, patterns in PROFICIENCY_PATTERNS.items():
            for p in patterns:
                if re.search(p, text_lower):
                    weight = max(weight, w)  # 取最高匹配
                    break
        
        # 清理熟练度描述词，保留技能名
        clean_text = text
        for patterns in PROFICIENCY_PATTERNS.values():
            for p in patterns:
                clean_text = re.sub(p, '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'[（(][^）)]*[）)]', '', clean_text)  # 去掉括号内容
        clean_text = clean_text.strip()
        
        return clean_text, weight
    
    # 1. 从技能列表提取（基础权重 0.7，根据描述调整）
    skill_list = resume.get("skills", [])
    if isinstance(skill_list, list):
        for s in skill_list:
            if s:
                skill_name, weight = extract_skill_and_weight(str(s).strip())
                if skill_name and len(skill_name) >= 2:
                    skills_weight[skill_name.lower()] = max(
                        skills_weight.get(skill_name.lower(), 0), weight
                    )
    elif isinstance(skill_list, str) and skill_list.strip():
        for part in re.split(r"[,，、;；/\s]+", skill_list.strip()):
            if part.strip():
                skill_name, weight = extract_skill_and_weight(part.strip())
                if skill_name and len(skill_name) >= 2:
                    skills_weight[skill_name.lower()] = max(
                        skills_weight.get(skill_name.lower(), 0), weight
                    )
    
    # 2. 从项目经历提取（项目中的技能权重更高，视为实际应用）
    # 使用通用技能词提取（非硬编码），覆盖所有行业
    for proj in (resume.get("project_exp") or [])[:3]:
        if isinstance(proj, dict):
            content = (proj.get("content", "") or "") + " " + (proj.get("title", "") or "")
            role = proj.get("role", "")
            
            # 通用技能词提取模式（覆盖所有行业，非硬编码特定技术）
            # 模式1: 驼峰命名技术词 (ReactNative, SpringBoot)
            # 模式2: 全大写缩写 (AWS, NLP, CAD)
            # 模式3: 中文技能词 (数据分析, 项目管理)
            # 模式4: 版本号技能 (Python3, ES6)
            skill_patterns = [
                r'\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b',  # 驼峰命名
                r'\b[A-Z]{2,}\b',  # 全大写缩写（2字母以上）
                r'[\u4e00-\u9fa5]{2,}(?:系统|平台|工具|软件|技术|框架|库|引擎|模型|算法|分析|设计|管理|开发|测试|运维|运营|营销)',  # 中文技能后缀
                r'\b(?:Python|Java|Go|Rust|C\+\+|JavaScript|TypeScript|PHP|Ruby|Swift|Kotlin)[\s\d\.]*\b',  # 编程语言（带版本号）
            ]
            
            extracted_skills = set()
            for pattern in skill_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                extracted_skills.update(m.strip() for m in matches if len(m.strip()) >= 2)
            
            # 同时提取已知的技能列表中的词（如果有的话）
            for skill in skill_list:
                if skill and skill.lower() in content.lower():
                    extracted_skills.add(skill.strip())
            
            for skill_key in extracted_skills:
                skill_key = skill_key.lower()
                # 基础权重：项目经历默认 0.85
                base_weight = 0.85
                
                # 项目负责人/核心开发者权重更高
                role_boost = 1.0
                if role and any(kw in role.lower() for kw in ['负责', '主导', '核心', '架构', 'lead', 'owner', '主程']):
                    role_boost = 1.2
                
                # 根据内容长度和详细程度调整（描述越详细，技能越可信）
                detail_boost = min(1.0, 0.8 + len(content) / 1000)
                
                final_weight = min(1.0, base_weight * role_boost * detail_boost)
                
                skills_weight[skill_key] = max(
                    skills_weight.get(skill_key, 0), final_weight
                )
    
    # 3. 从实习/工作经历提取（工作中的技能权重最高）
    # 同样使用通用技能词提取，非硬编码
    for work in (resume.get("work_exp") or [])[:2]:
        if isinstance(work, dict):
            content = (work.get("content", "") or "") + " " + (work.get("title", "") or "")
            
            # 工作描述中的技能视为核心技能（使用相同的通用提取模式）
            skill_patterns = [
                r'\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b',
                r'\b[A-Z]{2,}\b',
                r'[\u4e00-\u9fa5]{2,}(?:系统|平台|工具|软件|技术|框架|库|引擎|模型|算法|分析|设计|管理|开发|测试|运维|运营|营销)',
                r'\b(?:Python|Java|Go|Rust|C\+\+|JavaScript|TypeScript|PHP|Ruby|Swift|Kotlin)[\s\d\.]*\b',
            ]
            
            extracted_skills = set()
            for pattern in skill_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                extracted_skills.update(m.strip() for m in matches if len(m.strip()) >= 2)
            
            # 同时匹配技能列表中的词
            for skill in skill_list:
                if skill and skill.lower() in content.lower():
                    extracted_skills.add(skill.strip())
            
            for skill_key in extracted_skills:
                skill_key = skill_key.lower()
                # 工作经历中的技能默认最高权重 1.0
                skills_weight[skill_key] = max(skills_weight.get(skill_key, 0), 1.0)
    
    # 4. 从专业/课程提取（基础权重 0.4，作为了解级别）
    basic = resume.get("basic_info", {}) or {}
    major = basic.get("major", "")
    if major:
        major_key = str(major).strip().lower()
        skills_weight[major_key] = max(skills_weight.get(major_key, 0), 0.4)
    
    for edu in (resume.get("edu_bg") or [])[:1]:
        if isinstance(edu, dict) and edu.get("content"):
            courses = re.findall(r"[《\[]([^》\]]+)[》\]]", edu["content"])
            for c in courses:
                course_key = c.strip().lower()
                if len(course_key) >= 2:
                    skills_weight[course_key] = max(skills_weight.get(course_key, 0), 0.4)
    
    return skills_weight


def _extract_certificates_from_skills(skills: Dict[str, float]) -> set:
    """从带权重的技能字典中提取证书"""
    certs = set()
    cert_keywords = ["cet-4", "cet-6", "英语四级", "英语六级", "普通话", "计算机一级",
                     "计算机二级", "驾驶证", "会计证", "初级会计", "中级会计", "cpa",
                     "雅思", "托福", "gre", "gmat", "bec", "catti", "pmp", "教师资格证"]
    for s in skills.keys():
        for ck in cert_keywords:
            if ck in s:
                certs.add(ck)
    return certs


def _match_certificates(job_desc: str, certs: set) -> float:
    if not certs or not job_desc:
        return 0.0
    jd = job_desc.lower()
    score = 0.0
    if any(k in jd for k in ["cet-4", "cet4", "英语四级", "四级"]) and any(k in certs for k in ["cet-4", "英语四级"]):
        score += 0.06
    if any(k in jd for k in ["cet-6", "cet6", "英语六级", "六级"]) and any(k in certs for k in ["cet-6", "英语六级"]):
        score += 0.06
    if "普通话" in jd and any("普通话" in c for c in certs):
        score += 0.04
    if any(k in jd for k in ["计算机一级", "计算机二级", "计算机等级"]) and any("计算机" in c for c in certs):
        score += 0.04
    if any(k in jd for k in ["驾照", "驾驶证", "c1", "c2"]) and any("驾驶证" in c for c in certs):
        score += 0.03
    if any(k in jd for k in ["会计证", "初级会计", "cpa", "会计从业"]) and any(k in certs for k in ["会计证", "初级会计", "cpa"]):
        score += 0.05
    return min(score, 0.12)


def _award_score(resume: Dict[str, Any]) -> float:
    awards = resume.get("award", [])
    if not isinstance(awards, list) or not awards:
        return 0.0
    score = 0.0
    has_scholarship = False
    has_national = False
    has_provincial = False
    for a in awards:
        text = str(a).lower()
        if any(k in text for k in ["奖学金", "国家奖学金", "励志奖学金", "一等奖学金", "二等奖学金", "三等奖学金"]):
            has_scholarship = True
        if any(k in text for k in ["国家级", "全国", "国家奖", "一等奖", "金奖", "特等奖"]):
            has_national = True
        if any(k in text for k in ["省级", "省赛", "省一等奖", "省二等奖", "赛区"]):
            has_provincial = True
    if has_scholarship:
        score += 0.03
    if has_national:
        score += 0.03
    if has_provincial:
        score += 0.02
    score += min(0.01 * len(awards), 0.03)
    return min(score, 0.08)


def _extract_major_ngrams(text: str) -> set:
    """
    从专业名提取2-3字的n-gram子词，用于与岗位文本做模糊匹配。
    例：'机械工程' → {'机械', '械工', '工程', '机械工', '械工程'}
    配合jieba时额外加入分词结果，提升语义准确度。
    """
    tokens = set()
    # 提取2字和3字 n-gram
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            # 只保留纯中文或纯英文的gram，过滤标点/数字噪音
            if re.match(r'^[\u4e00-\u9fa5]+$', gram) or re.match(r'^[a-z]{2,}$', gram):
                tokens.add(gram)
    # 如果jieba可用，额外加入分词结果（通常比n-gram更准确）
    if JIEBA_AVAILABLE:
        for tok in jieba.lcut(text):
            if len(tok) >= 2:
                tokens.add(tok)
    return tokens


def _calc_major_match(major: str, job_name: str, job_desc: str,
                      major_vec_sim: float = -1.0) -> float:
    """
    通用专业匹配（不依赖硬编码专业枚举，覆盖各行各业）

    优先路径（当 major_vec_sim >= 0 时）：
        使用预计算的 text-embedding-v4 向量余弦相似度直接映射得分。
        向量语义匹配天然覆盖所有专业/岗位，无需维护任何白名单。
        sim >= 0.80 → 0.15（语义高度匹配，如"财务管理"→"财务会计"）
        sim >= 0.65 → 0.12（语义相关，如"公共事业管理"→"行政助理"）
        sim >= 0.50 → 0.09（语义弱相关，如"汉语言文学"→"文案策划"）
        sim >= 0.38 → 0.05（语义远相关，给少量基础分）
        sim <  0.38 → 0.0

    降级路径（major_vec_sim < 0，即向量化失败时）：
        依次使用字符串精确命中 → n-gram子词命中率 → 单字根 → Jaccard 四层兜底。
    """
    if not major:
        return 0.0

    # ── 向量语义路径（主路径）──
    if major_vec_sim >= 0.0:
        if major_vec_sim >= 0.80:
            return 0.15
        elif major_vec_sim >= 0.65:
            return 0.12
        elif major_vec_sim >= 0.50:
            return 0.09
        elif major_vec_sim >= 0.38:
            return 0.05
        else:
            return 0.0

    # ── 字符串降级路径（向量化失败时兜底）──
    job_text = (job_name + " " + job_desc).lower()

    # 第一层：精确命中
    if major in job_text:
        return 0.15

    # 第二层：n-gram子词命中率
    major_ngrams = _extract_major_ngrams(major)
    if not major_ngrams:
        return 0.0

    hit_ngrams = {t for t in major_ngrams if t in job_text}
    hit_ratio = len(hit_ngrams) / len(major_ngrams) if major_ngrams else 0.0

    if hit_ratio >= 0.6:
        return 0.13
    elif hit_ratio >= 0.4:
        return 0.10
    elif hit_ratio > 0:
        return 0.07

    # 第三层：单字根兜底
    major_roots = set(re.findall(r'[\u4e00-\u9fa5]{2,}', major))
    for root in major_roots:
        for seg in ([root] + ([root[:2]] if len(root) > 2 else [])):
            if seg in job_text:
                return 0.06 if seg in job_name.lower() else 0.03

    # 第四层：Jaccard兜底（专业名 vs 岗位名）
    job_name_ngrams = _extract_major_ngrams(job_name.lower())
    if major_ngrams and job_name_ngrams:
        intersection = major_ngrams & job_name_ngrams
        union = major_ngrams | job_name_ngrams
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard >= 0.3:
            return 0.06
        elif jaccard > 0:
            return 0.03

    return 0.0


def _calc_domain_match(skill_vec_sim: float) -> float:
    """
    领域/技能语义匹配得分（0-0.20）。
    
    由 calc_delivery_score 预计算「简历技能文本 vs JD」向量余弦相似度，
    通过 _meta_skill_vec_sim 注入 candidate。
    本函数仅做阈值映射，不再依赖任何硬编码领域字典，天然覆盖各行各业。
    
    Args:
        skill_vec_sim: 预计算的向量余弦相似度（-1 表示未计算）
    """
    if skill_vec_sim < 0:
        return 0.0
    if skill_vec_sim >= 0.85:
        return 0.20   # 技能高度匹配，如"Python/PyTorch"→"算法工程师"
    if skill_vec_sim >= 0.72:
        return 0.16
    if skill_vec_sim >= 0.60:
        return 0.12
    if skill_vec_sim >= 0.48:
        return 0.08
    if skill_vec_sim >= 0.38:
        return 0.04
    return 0.0


def _calc_experience_match(resume: Dict[str, Any], jd_text: str) -> float:
    """经历文本匹配：基于 project_exp + edu_bg 与 JD 的中英文词重叠度（Jaccard）
    
    使用 jieba 分词提升中文分词精度，若 jieba 不可用则降级到正则匹配
    """
    if not jd_text:
        return 0.0
    texts = []
    for proj in (resume.get("project_exp") or [])[:3]:
        if isinstance(proj, dict) and proj.get("content"):
            texts.append(proj["content"])
    for edu in (resume.get("edu_bg") or [])[:1]:
        if isinstance(edu, dict) and edu.get("content"):
            texts.append(edu["content"])
    if not texts:
        return 0.0

    all_text = " ".join(texts)
    
    # 提取简历文本词集合
    if JIEBA_AVAILABLE and jieba:
        # 使用 jieba 分词：提取2字以上的中文词和3字母以上的英文词
        resume_words = set()
        for word in jieba.lcut(all_text.lower()):
            word = word.strip()
            if len(word) >= 2:
                resume_words.add(word)
        # 额外提取英文词（jieba对英文支持较差）
        resume_words.update(set(re.findall(r'[a-z]{3,}', all_text.lower())))
    else:
        # 降级：使用正则匹配
        resume_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', all_text.lower()))
        resume_words.update(set(re.findall(r'[a-z]{3,}', all_text.lower())))

    # 提取JD文本词集合
    if JIEBA_AVAILABLE and jieba:
        jd_words = set()
        for word in jieba.lcut(jd_text.lower()):
            word = word.strip()
            if len(word) >= 2:
                jd_words.add(word)
        jd_words.update(set(re.findall(r'[a-z]{3,}', jd_text.lower())))
    else:
        jd_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', jd_text.lower()))
        jd_words.update(set(re.findall(r'[a-z]{3,}', jd_text.lower())))

    if not jd_words or not resume_words:
        return 0.0

    common = resume_words & jd_words
    union = resume_words | jd_words
    jaccard = len(common) / len(union) if union else 0.0
    return min(jaccard * 2.0, 0.15)


def _calc_city_score(work_city: str, job_cities: Optional[List[str]]) -> float:
    """
    城市匹配打分 - 增强版（支持城市群、远程匹配）
    
    匹配等级：
    - 精确匹配（同城市）：+0.10
    - 城市群匹配（长三角/珠三角等）：+0.06
    - 同省匹配：+0.03
    - 接受远程/异地：+0.05（特殊标记）
    - 不匹配：-0.05
    - work_city 为空：0（不参与）
    """
    if not work_city or not work_city.strip():
        return 0.0
    if not job_cities:
        return 0.0

    # 城市群定义（主要经济区域）
    CITY_GROUPS = {
        "长三角": ["上海", "南京", "苏州", "杭州", "宁波", "无锡", "常州", "南通", "嘉兴", "湖州", "绍兴", "金华", "台州", "合肥", "芜湖"],
        "珠三角": ["广州", "深圳", "佛山", "东莞", "中山", "珠海", "惠州", "江门", "肇庆"],
        "京津冀": ["北京", "天津", "石家庄", "唐山", "保定", "廊坊"],
        "成渝": ["成都", "重庆", "绵阳", "德阳", "眉山"],
        "武汉都市圈": ["武汉", "黄石", "鄂州", "黄冈", "孝感", "咸宁"],
        "西安都市圈": ["西安", "咸阳", "渭南", "铜川"],
    }
    
    # 省份映射（主要城市->省份）
    CITY_TO_PROVINCE = {
        "北京": "北京", "上海": "上海", "天津": "天津", "重庆": "重庆",
        "广州": "广东", "深圳": "广东", "佛山": "广东", "东莞": "广东",
        "南京": "江苏", "苏州": "江苏", "杭州": "浙江", "宁波": "浙江",
        "成都": "四川", "武汉": "湖北", "西安": "陕西",
        "郑州": "河南", "长沙": "湖南", "青岛": "山东", "济南": "山东",
        "沈阳": "辽宁", "大连": "辽宁", "哈尔滨": "黑龙江", "长春": "吉林",
        "昆明": "云南", "贵阳": "贵州", "南宁": "广西", "海口": "海南",
        "兰州": "甘肃", "银川": "宁夏", "西宁": "青海", "乌鲁木齐": "新疆",
        "拉萨": "西藏", "呼和浩特": "内蒙古", "太原": "山西", "石家庄": "河北",
        "合肥": "安徽", "南昌": "江西", "福州": "福建", "厦门": "福建",
    }
    
    # 远程工作关键词
    REMOTE_KEYWORDS = ["远程", "居家办公", "remote", "不限", "全国", "异地", "可接受"]

    # 统一处理：去除空格，支持逗号分隔的 work_city
    resume_cities = [c.strip() for c in work_city.split(",") if c.strip()]
    job_cities_norm = [c.strip() for c in job_cities if c and c.strip()]
    
    # 检查是否接受远程
    for rc in resume_cities:
        if any(kw in rc.lower() for kw in REMOTE_KEYWORDS):
            # 如果岗位也接受远程，给予正向分数
            for jc in job_cities_norm:
                if any(kw in jc.lower() for kw in REMOTE_KEYWORDS):
                    return 0.08
            # 简历接受远程但岗位不明确，给予中性分数
            return 0.03

    best_match = -0.05  # 默认不匹配
    
    for rc in resume_cities:
        rc_clean = rc.replace("市", "").replace("区", "")
        rc_province = CITY_TO_PROVINCE.get(rc.replace("市", ""), "")
        
        # 确定简历城市所属城市群
        rc_groups = set()
        for group_name, cities in CITY_GROUPS.items():
            if any(rc_clean in c or c in rc_clean for c in cities):
                rc_groups.add(group_name)
        
        for jc in job_cities_norm:
            jc_clean = jc.replace("市", "").replace("区", "")
            jc_province = CITY_TO_PROVINCE.get(jc.replace("市", ""), "")
            
            # 1. 精确匹配（同城市）
            if rc in jc or jc in rc or rc_clean == jc_clean:
                best_match = max(best_match, 0.10)
                continue
            
            # 2. 城市群匹配
            for group_name, cities in CITY_GROUPS.items():
                rc_in_group = any(rc_clean in c or c in rc_clean for c in cities)
                jc_in_group = any(jc_clean in c or c in jc_clean for c in cities)
                if rc_in_group and jc_in_group:
                    best_match = max(best_match, 0.06)
                    break
            
            # 3. 同省匹配
            if rc_province and jc_province and rc_province == jc_province:
                best_match = max(best_match, 0.03)
    
    return best_match


# ----------------------------- 投递专用重排器 -----------------------------
class AttachmentDeliveryReranker:
    def __init__(self, rerank_client: AsyncRerankClient):
        self.rerank_client = rerank_client

    async def rerank(
            self,
            resume: Dict[str, Any] = None,
            candidates: List[Dict] = None,
            job_titles: List[str] = None,
            job_info: List[Dict] = None,
            resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
            intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT,
            top_k: int = 100,
    ) -> List[Dict]:
        if not candidates:
            return []
        coarse_scored = self._coarse_rank(
            resume, candidates, job_titles, job_info, resume_weight, intent_weight
        )
        return await self._fine_rank(
            resume, coarse_scored, job_titles, job_info, resume_weight, intent_weight, top_k
        )

    def _coarse_rank(
            self,
            resume: Dict[str, Any],
            candidates: List[Dict],
            job_titles: List[str] = None,
            job_info: List[Dict] = None,
            resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
            intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT,
    ) -> List[Dict]:
        if not resume:
            return candidates[:150]

        basic = resume.get("basic_info", {}) or {}
        resume_skills = _extract_resume_skills(resume)
        resume_intents = [str(i).strip().lower() for i in (basic.get("intent") or []) if i]
        resume_degree = _parse_degree_level(basic.get("degree", ""))
        certs = _extract_certificates_from_skills(resume_skills)
        is_top_school_flag = _is_top_school(resume)
        award_score = _award_score(resume)

        # work_city 需要从 candidate 的 meta 里取（我们在构造 candidate 时塞进去）
        work_city = ""
        if candidates and candidates[0].get("_meta_work_city") is not None:
            work_city = candidates[0].get("_meta_work_city", "")
        job_cities = candidates[0].get("_meta_job_cities") if candidates else None

        scored = []
        for cand in candidates:
            job_desc = (cand.get("job_describe", "") or "").lower()
            job_name = (cand.get("job_name", "") or "").strip()

            resume_score = 0.0
            resume_features = {}

            # 1. 技能语义匹配（0-0.55）- 权重提高，增加区分度
            # 使用 calc_delivery_score 预计算的「简历技能文本 vs JD」向量余弦相似度。
            # 相比原先的字符串逐词查找+字典扩展，向量方案天然覆盖各行各业，
            # 不需要维护任何技能白名单或同义词表。
            skill_vec_sim = cand.get("_meta_skill_vec_sim", -1.0)
            if skill_vec_sim >= 0:
                # 向量路径：相似度 → [0, 1] 标准化后乘权重
                # 使用更激进的分段映射，低相似度给低分，高相似度给高分
                if skill_vec_sim >= 0.80:
                    skill_score = 1.0  # 高度匹配
                elif skill_vec_sim >= 0.65:
                    skill_score = 0.75 + (skill_vec_sim - 0.65) / (0.80 - 0.65) * 0.25
                elif skill_vec_sim >= 0.50:
                    skill_score = 0.45 + (skill_vec_sim - 0.50) / (0.65 - 0.50) * 0.30
                elif skill_vec_sim >= 0.35:
                    skill_score = 0.20 + (skill_vec_sim - 0.35) / (0.50 - 0.35) * 0.25
                else:
                    skill_score = max(0.0, skill_vec_sim / 0.35 * 0.20)
                
                # 硬性门槛：技能相似度低于0.25视为完全不匹配，给予惩罚
                if skill_vec_sim < 0.25:
                    skill_score = skill_score * 0.3  # 大幅降权
                    resume_features["skill_mismatch_penalty"] = True
                
                resume_score += skill_score * 0.55  # 权重从0.45提高到0.55
                resume_features["skill_match"] = round(skill_score, 2)
                resume_features["skill_vec_sim"] = round(skill_vec_sim, 4)
            elif resume_skills and job_desc:
                # 降级路径：向量化失败时用加权词重叠
                # 使用技能权重：核心技能(1.0)命中 > 熟练技能(0.7) > 了解技能(0.4)
                weighted_match = 0.0
                total_weight = 0.0
                for skill, weight in resume_skills.items():
                    total_weight += weight
                    if skill.lower() in job_desc:
                        weighted_match += weight
                
                # 加权匹配率，同时考虑技能覆盖度和权重集中度
                if total_weight > 0:
                    coverage = weighted_match / total_weight  # 加权覆盖率
                    # 额外奖励：高权重技能命中给予 bonus
                    core_hits = sum(1 for s, w in resume_skills.items() if w >= 0.9 and s.lower() in job_desc)
                    bonus = min(0.2, core_hits * 0.1)  # 每个核心技能命中最多+0.1
                    skill_score = min(1.0, coverage * 0.8 + bonus)
                else:
                    skill_score = 0.0
                
                resume_score += skill_score * 0.45
                resume_features["skill_match"] = round(skill_score, 2)
                resume_features["skill_match_fallback"] = True
                resume_features["skill_weighted_coverage"] = round(weighted_match / total_weight, 2) if total_weight > 0 else 0

            # 2. 意向匹配（0-0.25）
            intent_score = 0.0
            intent_features = {}
            if resume_intents:
                intent_match = self._calc_intent_match(job_name, resume_intents)
                intent_score += intent_match * 0.25
                intent_features["intent_title_match"] = round(intent_match, 2)

            # 3. 专业匹配（0-0.20）- 权重提高，增加硬性门槛
            # 优先使用 calc_delivery_score 预计算的向量余弦相似度（语义级，覆盖所有专业）；
            # 向量化失败时（major_vec_sim < 0）自动降级为字符串 n-gram 兜底。
            major = (basic.get("major") or "").lower().strip()
            major_vec_sim = cand.get("_meta_major_vec_sim", -1.0)
            major_score = _calc_major_match(major, job_name, job_desc,
                                            major_vec_sim=major_vec_sim)
            
            # 硬性门槛：专业相似度极低时视为不匹配，给予惩罚
            if major_vec_sim >= 0 and major_vec_sim < 0.30 and major:
                major_score = major_score * 0.3  # 大幅降权
                resume_features["major_mismatch_penalty"] = True
            
            resume_score += major_score * 1.33  # 原最大0.15，现在最大约0.20
            resume_features["major_match"] = round(major_score, 2)
            if major_vec_sim >= 0:
                resume_features["major_vec_sim"] = round(major_vec_sim, 4)

            # 4. 领域/技能语义匹配（0-0.15）- 权重降低，避免重复计分
            # 注意：domain_match 现在复用 skill_vec_sim（同一组向量的不同阈值映射），
            # 代表简历技能与岗位领域的整体语义契合度，不再依赖硬编码领域字典。
            domain_score = _calc_domain_match(skill_vec_sim)
            # 如果技能已经高度匹配，领域分适当降低避免重复奖励
            if skill_vec_sim >= 0.70:
                domain_score = domain_score * 0.6
            resume_score += domain_score * 0.75  # 原最大0.20，现在最大约0.15
            resume_features["domain_match"] = round(domain_score, 2)

            # 5. 经历文本匹配（0-0.15）
            exp_score = _calc_experience_match(resume, cand.get("job_describe", ""))
            resume_score += exp_score
            resume_features["experience_match"] = round(exp_score, 2)

            # 6. 学历匹配（0-0.15）- 增加硬性门槛惩罚
            jd_degree_req = _extract_degree_requirement(cand.get("job_describe", ""))
            degree_score = 0.0
            if resume_degree > 0 and jd_degree_req > 0:
                if resume_degree >= jd_degree_req:
                    degree_score = 0.12 + 0.03 * (resume_degree - jd_degree_req)
                else:
                    # 学历不达标：大幅降权，不是简单给低分
                    degree_score = -0.10  # 负分惩罚
                    resume_features["degree_mismatch_penalty"] = True
            elif resume_degree > 0:
                degree_score = 0.06 + 0.02 * resume_degree
            degree_score = min(degree_score, 0.15)
            resume_score += degree_score
            resume_features["degree_match"] = round(degree_score, 2)

            # 7. 院校等级（0-0.1）
            if is_top_school_flag:
                school_score = 0.08
                if any(k in job_desc for k in ["985", "211", "双一流", "重点院校", "知名高校", "重点大学"]):
                    school_score += 0.04
                resume_score += min(school_score, 0.12)
                resume_features["top_school"] = True

            # 8. 证书匹配（0-0.12）
            cert_score = _match_certificates(cand.get("job_describe", ""), certs)
            resume_score += cert_score
            resume_features["cert_match"] = round(cert_score, 2)

            # 9. 奖项（0-0.08）
            resume_score += award_score
            resume_features["award_score"] = round(award_score, 2)

            # 10. 城市匹配（-0.05 ~ +0.08）
            city_score = _calc_city_score(work_city, job_cities)
            intent_score += city_score
            resume_features["city_score"] = round(city_score, 2)

            # 兜底：RRF（投递场景通常为 0）
            resume_score += min(cand.get("rrf_score", 0) * 0.3, 0.3)

            total_weight = resume_weight + intent_weight or 1.0
            final_c_score = (
                    resume_score * (resume_weight / total_weight) +
                    intent_score * (intent_weight / total_weight)
            )

            cand.update({
                "coarse_score": round(final_c_score, 4),
                "resume_score": round(resume_score, 4),
                "intent_score": round(intent_score, 4),
                "resume_features": resume_features,
                "intent_features": intent_features,
            })
            scored.append(cand)

        scored.sort(key=lambda x: x["coarse_score"], reverse=True)
        filtered = [c for c in scored if c["coarse_score"] >= THRESHOLD.COARSE_MIN_SCORE]
        if not filtered:
            filtered = scored[:50]
        return filtered[:150]

    @staticmethod
    def _calc_intent_match(job_name: str, intents: List[str]) -> float:
        if not job_name or not intents:
            return 0.0
        job_lower = job_name.lower()
        best = 0.0
        for intent in intents:
            intent_lower = intent.lower()
            if job_lower == intent_lower:
                return 1.0
            if job_lower in intent_lower or intent_lower in job_lower:
                best = max(best, 0.8)
                continue
            job_parts = set(re.split(r"[\s\/\,，、;；]+", job_lower))
            intent_parts = set(re.split(r"[\s\/\,，、;；]+", intent_lower))
            if job_parts and intent_parts:
                common = job_parts & intent_parts
                union = job_parts | intent_parts
                if union:
                    jaccard = len(common) / len(union)
                    if jaccard >= 0.5:
                        best = max(best, 0.6)
                    elif jaccard >= 0.3:
                        best = max(best, 0.35)
        return best

    async def _fine_rank(
            self,
            resume: Dict[str, Any],
            candidates: List[Dict],
            job_titles: List[str] = None,
            job_info: List[Dict] = None,
            resume_weight: float = RERANK_RESUME_WEIGHT_DEFAULT,
            intent_weight: float = RERANK_INTENT_WEIGHT_DEFAULT,
            top_k: int = 100,
    ) -> List[Dict]:
        if not candidates:
            return []

        query = RerankQueryBuilder.build_query(
            resume=resume,
            job_titles=job_titles,
            job_info=job_info,
            resume_weight=resume_weight,
            intent_weight=intent_weight,
        )
        if not query:
            return candidates[:top_k]

        documents = []
        for cand in candidates:
            job_desc = cand.get("job_describe", "") or ""
            desc_part = job_desc[:500] if len(job_desc) > 500 else job_desc
            doc_parts = [
                f"岗位：{cand.get('job_name', '')}",
                f"描述与要求：{desc_part}",
            ]
            documents.append("\n".join(doc_parts))

        rerank_results = await self.rerank_client.rerank(query, documents, top_k=len(candidates))
        if not rerank_results:
            return candidates[:top_k]

        # 适配新的返回格式：优先使用 normalized_score，fallback 时使用 score
        raw_rerank_map = {
            r["index"]: r.get("normalized_score", r.get("score", 0)) 
            for r in rerank_results
        }
        # 标记是否为兜底结果
        fallback_map = {r["index"]: r.get("is_fallback", False) for r in rerank_results}
        
        # 精排分数校准（激进版）：对LLM的保守打分进行强非线性映射
        # 目标：低分(<0.5)大幅降低，高分(>0.75)大幅提升
        rerank_map = {}
        for idx, raw_score in raw_rerank_map.items():
            if fallback_map.get(idx, False):
                # 兜底结果：不做校准，但限制在0.3-0.7之间
                rerank_map[idx] = max(0.3, min(0.7, raw_score))
            else:
                # 非线性校准：更激进的映射
                if raw_score >= 0.80:
                    # 高分段：0.80->0.92, 1.0->1.0
                    calibrated = 0.92 + (raw_score - 0.80) / 0.20 * 0.08
                elif raw_score >= 0.65:
                    # 中高分段：0.65->0.72, 0.80->0.92
                    calibrated = 0.72 + (raw_score - 0.65) / 0.15 * 0.20
                elif raw_score >= 0.50:
                    # 中分段：0.50->0.55, 0.65->0.72
                    calibrated = 0.55 + (raw_score - 0.50) / 0.15 * 0.17
                elif raw_score >= 0.35:
                    # 中低分段：0.35->0.30, 0.50->0.55（低分反而降）
                    calibrated = 0.30 + (raw_score - 0.35) / 0.15 * 0.25
                else:
                    # 极低分段：大幅降低
                    calibrated = max(0.05, raw_score * 0.5)
                rerank_map[idx] = calibrated

        final_scored = []
        for i, cand in enumerate(candidates):
            rerank_score = rerank_map.get(i, 0)
            coarse_score = cand.get("coarse_score", 0)
            is_fallback = fallback_map.get(i, False)
            
            # 投递场景使用自定义融合权重
            raw_final_score = coarse_score * DELIVERY_COARSE_WEIGHT + rerank_score * DELIVERY_RERANK_WEIGHT
            
            # 最终分数校准（参考 job_recommend_campus 的 min-max 思路）
            # 由于只有单 candidate，这里基于粗排和精排的相对差异做校准
            # 核心思想：粗排和精排都高才给高分，有一个低就降分
            
            # 计算粗排和精排的"一致性"：越高一致性越高，最终分数越高
            avg_score = (coarse_score + rerank_score) / 2
            score_gap = abs(coarse_score - rerank_score)
            
            # 一致性奖励：两个分数越接近，奖励越高
            consistency_bonus = max(0, 0.15 - score_gap * 0.3)
            
            # 最终分数 = 加权平均 + 一致性奖励
            final_score = min(1.0, raw_final_score + consistency_bonus)
            
            # 硬性门槛：粗排或精排任意一个低于0.3，直接降权
            if coarse_score < 0.3 or rerank_score < 0.3:
                final_score = final_score * 0.6  # 强制降权
            
            # 极度不匹配惩罚：两个都低于0.4
            if coarse_score < 0.4 and rerank_score < 0.4:
                final_score = min(final_score, 0.35)  # 强制不超过0.35
            
            # 极度匹配奖励：两个都高于0.8
            if coarse_score >= 0.8 and rerank_score >= 0.8:
                final_score = max(final_score, 0.90)  # 强制不低于0.90

            cand["final_score"] = round(final_score, 4)
            cand["rerank_score"] = round(rerank_score, 4)
            cand["rerank_fallback"] = is_fallback  # 标记是否为兜底结果
            final_scored.append(cand)

        filtered = [c for c in final_scored if c["final_score"] >= THRESHOLD.RERANK_MIN_SCORE]
        if not filtered:
            filtered = final_scored[:min(20, len(final_scored))]

        filtered.sort(key=lambda x: x["final_score"], reverse=True)
        return filtered[:top_k]


# ----------------------------- 业务逻辑 -----------------------------
async def calc_delivery_score(
        resume_data: Dict[str, Any],
        work_city: str,
        job_name: str,
        job_describe: str,
        job_cities: Optional[List[str]] = None,
        use_intent: bool = True,
) -> float:
    """
    计算简历与岗位的匹配分数（多维度向量化优化版）
    
    【核心优化】多维度技能向量化：显性技能 + 项目经历 + 工作经历，取Max相似度
    
    Args:
        resume_data: 简历解析数据
        work_city: 期望工作城市
        job_name: 岗位名称
        job_describe: 岗位描述
        job_cities: 岗位所在城市列表
        use_intent: 是否启用意向匹配（默认启用，与推荐系统一致）
    """
    # ── 预计算向量相似度（batch 调用，一次 API 请求计算所有向量）──
    # 用 text-embedding-v4 计算：
    #   1. 专业名 vs 岗位文本（major_vec_sim）
    #   2. 多维度技能 vs 岗位文本（skill_vec_sim）：显性技能 + 项目经历 + 工作经历，取Max
    major_vec_sim = -1.0
    skill_vec_sim = -1.0
    try:
        basic = resume_data.get("basic_info", {}) or {}
        major = (basic.get("major") or "").strip()
        
        # 提取多维度技能文本
        resume_skills_raw = _extract_resume_skills(resume_data)
        project_text = _extract_project_text(resume_data)
        work_text = _extract_work_text(resume_data)

        # 岗位侧文本：岗位名 + JD前350字
        job_side_text = f"{job_name} {job_describe[:350]}"

        # 构建多维度技能文本列表
        skill_dimensions = []
        if resume_skills_raw:
            # 显性技能：按权重排序，优先取高权重技能
            sorted_skills = sorted(resume_skills_raw.items(), key=lambda x: x[1], reverse=True)
            high_weight = [s for s, w in sorted_skills if w >= 0.7][:20]
            med_weight = [s for s, w in sorted_skills if 0.4 <= w < 0.7][:10]
            selected_skills = high_weight + med_weight
            skill_dimensions.append(" ".join(selected_skills[:30]))
        if project_text:
            skill_dimensions.append(project_text)
        if work_text:
            skill_dimensions.append(work_text)

        # 构建 batch 请求
        texts_to_embed = []
        indices = {}  # 记录各维度在batch中的索引
        idx = 0

        # 专业
        if major and _state.embedding_client:
            indices['major'] = idx
            texts_to_embed.append(major)
            idx += 1

        # 技能各维度
        for i, dim_text in enumerate(skill_dimensions):
            if dim_text and _state.embedding_client:
                indices[f'skill_{i}'] = idx
                texts_to_embed.append(dim_text)
                idx += 1

            # 岗位侧文本（最后一个）
            indices['job'] = idx
            texts_to_embed.append(job_side_text)

            embeddings = await _state.embedding_client.embed_texts(texts_to_embed)
            job_vec = np.asarray(embeddings[indices['job']], dtype=np.float32) if embeddings[indices['job']] else None

            if job_vec is not None:
                # 专业相似度
                if 'major' in indices and embeddings[indices['major']] is not None:
                    major_vec_sim = float(cosine_sim(
                        np.asarray(embeddings[indices['major']], dtype=np.float32),
                        job_vec
                    ))
                    logger.debug(f"专业向量相似度: major={major!r}, job={job_name!r}, sim={major_vec_sim:.4f}")

                # 技能相似度（多维度取Max，确保覆盖度）
                skill_sims = []
                for key in indices:
                    if key.startswith('skill_') and embeddings[indices[key]] is not None:
                        sim = float(cosine_sim(
                            np.asarray(embeddings[indices[key]], dtype=np.float32),
                            job_vec
                        ))
                        skill_sims.append(sim)
                
                if skill_sims:
                    skill_vec_sim = max(skill_sims)  # 取最高匹配分
                    logger.debug(f"技能多维度匹配: dims={len(skill_sims)}, max_sim={skill_vec_sim:.3f}")

    except Exception as e:
        logger.warning(f"向量化预计算失败，降级到字符串匹配: {e}")

    candidate = {
        "job_id": 0,
        "job_name": job_name,
        "job_describe": job_describe,
        "city": "",
        # meta 字段供粗排使用
        "_meta_work_city": work_city,
        "_meta_job_cities": job_cities,
        "_meta_major_vec_sim": major_vec_sim,   # 专业语义相似度
        "_meta_skill_vec_sim": skill_vec_sim,   # 技能语义相似度（替代字典匹配）
    }
    
    # 意向权重：启用时为0.15，与推荐系统推荐的0.2略有调整
    # 投递场景意向权重稍低，避免简历意向填写模糊时误伤
    intent_weight = 0.15 if use_intent else 0.0
    resume_weight = 1.0 - intent_weight
    
    # 【优化】添加Dummy Candidates解决单candidate rerank打分偏低问题
    # 通过对比基准岗位，让LLM有更明确的参照系
    all_candidates = [candidate] + DUMMY_CANDIDATES
    
    results = await _state.delivery_reranker.rerank(
        resume=resume_data,
        candidates=all_candidates,
        job_titles=[job_name],  # 传入岗位名称用于意向匹配
        job_info=None,
        resume_weight=resume_weight,
        intent_weight=intent_weight,
        top_k=1,
    )
    if not results:
        return 0.0
    
    # 过滤掉dummy结果，只返回真实岗位的分数
    real_results = [r for r in results if not r.get("is_dummy", False)]
    if real_results:
        return round(real_results[0].get("final_score", 0.0), 4)
    return 0.0


async def _get_resume_data(resume_id: int, resume_type: int) -> Optional[Dict[str, Any]]:
    resume_item = await get_attachment_resume_by_id(_state.db_pool, resume_id, resume_type)
    if not resume_item:
        logger.warning(f"简历不存在: resume_id={resume_id}, resume_type={resume_type}")
        return None
    post_address = resume_item.get("post_address")
    if not post_address:
        logger.warning(f"简历解析数据为空: resume_id={resume_id}")
        return None
    try:
        if isinstance(post_address, str):
            resume_data = json.loads(post_address)
        else:
            resume_data = post_address
    except json.JSONDecodeError:
        logger.warning(f"简历 JSON 解析失败: resume_id={resume_id}")
        return None

    return {
        "resume_data": resume_data,
        "work_city": resume_item.get("work_city", ""),
    }


# ----------------------------- API 端点 -----------------------------
@app.post("/api/v1/jobs/delivery_score_campus", response_model=DeliverResponse)
async def delivery_score_campus(req: CampusDeliverRequest) -> DeliverResponse:
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_CAMPUS)
    if not info:
        return DeliverResponse(score=0.0)
    score = await calc_delivery_score(
        info["resume_data"], info["work_city"],
        req.job_name, req.job_describe, req.job_cities
    )
    logger.info(f"实习校招投递分数: resume_id={req.resume_id}, job={req.job_name}, score={score}")
    return DeliverResponse(score=score)


@app.post("/api/v1/jobs/delivery_score_part_time", response_model=DeliverResponse)
async def delivery_score_part_time(req: PartTimeDeliverRequest) -> DeliverResponse:
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_PART_TIME)
    if not info:
        return DeliverResponse(score=0.0)
    score = await calc_delivery_score(
        info["resume_data"], info["work_city"],
        req.job_name, req.job_describe, req.job_cities
    )
    logger.info(f"兼职投递分数: resume_id={req.resume_id}, job={req.job_name}, score={score}")
    return DeliverResponse(score=score)


@app.post("/api/v1/jobs/delivery_score_campus_by_job_id", response_model=DeliverResponse)
async def delivery_score_campus_by_job_id(req: CampusDeliverByJobIdRequest) -> DeliverResponse:
    """实习校招岗位投递匹配分数（通过 job_id 获取岗位详情）"""
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_CAMPUS)
    if not info:
        return DeliverResponse(score=0.0)

    job_data = await fetch_job_detail(req.job_id, req.api_token)
    if not job_data:
        logger.warning(f"无法获取岗位详情: job_id={req.job_id}")
        return DeliverResponse(score=0.0)

    parsed = parse_job_detail(job_data)
    job_cities = [parsed["city"]] if parsed.get("city") else None

    score = await calc_delivery_score(
        info["resume_data"], info["work_city"],
        parsed["job_name"], parsed["job_describe"], job_cities
    )
    logger.info(
        f"实习校招投递分数(by_job_id): resume_id={req.resume_id}, job_id={req.job_id}, job={parsed['job_name']}, score={score}")
    return DeliverResponse(score=score)


@app.post("/api/v1/jobs/delivery_score_part_time_by_job_id", response_model=DeliverResponse)
async def delivery_score_part_time_by_job_id(req: PartTimeDeliverByJobIdRequest) -> DeliverResponse:
    """兼职岗位投递匹配分数（通过 job_id 获取岗位详情）"""
    info = await _get_resume_data(req.resume_id, RESUME_TYPE_PART_TIME)
    if not info:
        return DeliverResponse(score=0.0)

    job_data = await fetch_job_detail(req.job_id, req.api_token)
    if not job_data:
        logger.warning(f"无法获取岗位详情: job_id={req.job_id}")
        return DeliverResponse(score=0.0)

    parsed = parse_job_detail(job_data)
    job_cities = [parsed["city"]] if parsed.get("city") else None

    score = await calc_delivery_score(
        info["resume_data"], info["work_city"],
        parsed["job_name"], parsed["job_describe"], job_cities
    )
    logger.info(
        f"兼职投递分数(by_job_id): resume_id={req.resume_id}, job_id={req.job_id}, job={parsed['job_name']}, score={score}")
    return DeliverResponse(score=score)


if __name__ == "__main__":
    import uvicorn
    from config import SERVICE_HOST

    uvicorn.run(app, host=SERVICE_HOST, port=8233, log_level="info")
