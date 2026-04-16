# -*- coding: utf-8 -*-
"""
配置文件 - 包含所有可配置的参数
"""
import os

# ----------------------------- 岗位详情 API 配置 -----------------------------
JOB_DETAIL_API_BASE = "https://apis.pocketuni.net/pu-app/api/v2/jobOpenings"
JOB_DETAIL_API_TOKEN = os.environ.get(
    "JOB_DETAIL_API_TOKEN",
    "d36bd74f37e311f1b01e00163e4451f0:201678164983808;"
)

# ----------------------------- Milvus 配置 -----------------------------
MILVUS_HOST = "192.168.0.2"
MILVUS_PORT = "19530"
COLLECTION_NAME = "job_recommendations"
EMBEDDING_DIM = 1024
VECTOR_DIM = EMBEDDING_DIM

# ----------------------------- 数据库配置 -----------------------------
DB_CONFIG = {
    "host": "192.168.0.218",
    "port": 3306,
    "user": "root",
    "password": "root123",
    "database": "pu_recruitment",
    "charset": "utf8mb4"
}

# ----------------------------- DashScope API 配置 -----------------------------
DASHSCOPE_API_KEY = "sk-b8855f2599ea4111a20e3b621713e97b"
QWEN_API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
EMBEDDING_API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'
RERANK_API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'

# ----------------------------- 并发控制配置 -----------------------------
EMBEDDING_CONCURRENCY = 5  # 向量化并发限制
MILVUS_CONCURRENCY = 3  # Milvus查询并发限制

# ----------------------------- 线程池配置 -----------------------------
THREAD_POOL_MAX_WORKERS = 16
THREAD_POOL_CPU_MULTIPLIER = 2

# ----------------------------- HTTP Session 配置 -----------------------------
HTTP_TIMEOUT_TOTAL = 30
HTTP_TIMEOUT_CONNECT = 5
HTTP_CONNECTOR_LIMIT = 20
HTTP_CONNECTOR_LIMIT_PER_HOST = 10

# ----------------------------- 本地嵌入缓存路径 -----------------------------
LOCAL_EMBEDDINGS_CACHE_PATH = r"/root/code/Job_Recommendations/hopejobs_qwen_v4_text_embeddings.json"


# ----------------------------- 阈值配置 -----------------------------
class ThresholdConfig:
    """阈值配置 - 用于过滤低质量匹配"""
    # 召回阶段阈值
    RECALL_MIN_SIM = 0.45  # 向量相似度最低阈值 (0-1)

    # 粗排阶段阈值
    COARSE_MIN_SCORE = 0.15  # 粗排最低分数 (0-1)

    # 精排阶段阈值
    RERANK_MIN_SCORE = 0.25  # 重排最低分数 (0-1)

    # 融合阶段阈值
    FUSION_MIN_SCORE = {
        "resume_only": 0.20,  # 仅简历召回的最低分
        "intent_only": 0.20,  # 仅意向召回的最低分
        "dual_channel": 0.35,  # 双通道召回的最低分 (要求更高)
    }

    # 双通道加成参数
    DUAL_CHANNEL_BONUS = 0.15  # 双通道加成系数
    SECONDARY_WEIGHT = 0.3  # 副通道权重


THRESHOLD = ThresholdConfig()

# ----------------------------- 简历字段权重配置 -----------------------------
RESUME_FIELD_WEIGHTS = {
    # 技能类（最高权重）
    "skills": 4.0,
    "skill_keywords": 4.0,
    "tech_stack": 3.5,

    # 经历类
    "project_exp": 3.0,
    "internship_exp": 3.0,
    "work_exp": 2.5,

    # 专业背景
    "major": 2.0,
    "courses": 1.5,

    # 求职意向
    "intent_position": 2.5,
    "intent_industry": 1.5,

    # 其他
    "certificates": 1.0,
    "awards": 0.8,
}

# 在线简历字段权重配置
ONLINE_RESUME_FIELD_WEIGHTS = {
    "skill": 4.0,
    "project_experience": 3.0,
    "work_experience": 3.0,
    "internship_experience": 3.0,
    "education_experience": 2.0,
    "profession": 2.0,
    "resume_title": 2.0,
}

# ----------------------------- 重排查询配置 -----------------------------
RERANK_QUERY_MAX_LENGTH = 600
RERANK_RESUME_WEIGHT_DEFAULT = 0.6
RERANK_INTENT_WEIGHT_DEFAULT = 0.4

# 粗排/精排分数融合权重
COARSE_SCORE_WEIGHT = 0.3
RERANK_SCORE_WEIGHT = 0.7

# ----------------------------- 搜索配置 -----------------------------
# RRF 融合参数
RRF_K_DEFAULT = 60

# 候选数量限制
CANDIDATE_LIMIT_DEFAULT = 500
TOP_K_DEFAULT = 300

# 分页默认参数
PAGE_NUM_DEFAULT = 1
PAGE_SIZE_DEFAULT = 20

# 重排截断倍数
RERANK_CUTOFF_MULTIPLIER = 2

# 最小相似度阈值（用于兼职搜索）
MIN_SIMILARITY_THRESHOLD = 0.35

# ----------------------------- 服务配置 -----------------------------
# 实习校招服务端口
CAMPUS_SERVICE_PORT = 8222

# 兼职服务端口
PART_TIME_SERVICE_PORT = 8123

# 服务主机
SERVICE_HOST = "0.0.0.0"

# ----------------------------- 简历类型配置 -----------------------------
# 简历类型
RESUME_TYPE_CAMPUS = 1  # 实习/校招
RESUME_TYPE_PART_TIME = 2  # 兼职

# 工作性质
WORK_TYPE_FULL_TIME = 1  # 全职
WORK_TYPE_INTERNSHIP = 2  # 实习
WORK_TYPE_PART_TIME = 3  # 兼职

# 默认工作性质（用于过滤）
WORK_NATURE_CAMPUS = 2  # 实习校招默认 workNature
WORK_NATURE_PART_TIME = 3  # 兼职默认 workNature
