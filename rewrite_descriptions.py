
import os
import csv
import time
import requests
from tqdm import tqdm

# ───────────────────────────────────────────────
# 配置
# ───────────────────────────────────────────────
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_API_KEY = "sk-b8855f2599ea4111a20e3b621713e97b"  
QWEN_MODEL   = "qwen-turbo"  

OUTPUT_DIR  = "data/aicard"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rewritten_descriptions.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


REQUEST_INTERVAL = 0.3

DB_CONFIG = {
    "host": "192.168.0.218",
    "port": 25236,
    "user": "SYSDBA",
    "password": "Dameng123",
}

# ───────────────────────────────────────────────
# 各类内容的提示词模板
# ───────────────────────────────────────────────
PROMPTS = {
    "shop": (
        "你是一个本地生活平台的内容编辑，请将以下店铺信息改写成一句简洁、口语化的中文描述，"
        "要求：说明店铺名称、类型、主营内容，15~30字，不要夸张宣传语。\n\n"
        "店铺名称：{name}\n"
        "原始描述：{description}\n"
        "地址：{address}\n\n"
        "改写后（只输出改写结果，不要其他内容）："
    ),
    "school": (
        "你是一个本地生活平台的内容编辑，请将以下夜校信息改写成一句简洁、口语化的中文描述，"
        "要求：说明夜校名称、主要教授的课程或技能，15~30字，不要夸张宣传语。\n\n"
        "夜校名称：{name}\n"
        "原始描述：{description}\n"
        "地址：{address}\n\n"
        "改写后（只输出改写结果，不要其他内容）："
    ),
    "course": (
        "你是一个本地生活平台的内容编辑，请将以下课程信息改写成一句简洁、口语化的中文描述，"
        "要求：说明课程名称、内容和适合人群，15~30字，不要夸张宣传语。\n\n"
        "课程名称：{name}\n"
        "原始描述：{description}\n"
        "地址：{address}\n\n"
        "改写后（只输出改写结果，不要其他内容）："
    ),
    "apartment": (
        "你是一个本地生活平台的内容编辑，请将以下公寓信息改写成一句简洁、口语化的中文描述，"
        "要求：说明公寓名称、类型、主要特点，15~30字，不要夸张宣传语。\n\n"
        "公寓名称：{name}\n"
        "原始描述：{description}\n"
        "地址：{address}\n\n"
        "改写后（只输出改写结果，不要其他内容）："
    ),
}

QUERIES = {
    "shop": """
        SELECT id, name, description, address
        FROM merchant_main
        WHERE is_open=1 AND is_use=1 AND status=1 AND no_recommend=0
          AND deleted_at IS NULL
        ORDER BY id
    """,
    "school": """
        SELECT id, name, service_summary AS description, address
        FROM campus
        WHERE is_use=1 AND status=1 AND deleted_at IS NULL
        ORDER BY id
    """,
    "course": """
        SELECT c.id, c.name, c.description, ca.address
        FROM course c
        JOIN campus ca ON c.campus_id = ca.id
        WHERE c.is_use=1 AND c.status=1
          AND ca.is_use=1 AND ca.status=1 AND ca.deleted_at IS NULL
        ORDER BY c.id
    """,
    "apartment": """
        SELECT am.id, am.name, am.description, am.address
        FROM apartment_main am
        WHERE am.is_open=1 AND am.is_use=1 AND am.status=1
          AND am.deleted_at IS NULL
        ORDER BY am.id
    """,
}


# ───────────────────────────────────────────────
# 数据库
# ───────────────────────────────────────────────
def get_db_conn():
    import dmPython
    return dmPython.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        server=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
    )


def query_db(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0].lower() for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return [dict(zip(cols, row)) for row in rows]


# ───────────────────────────────────────────────
# Qwen API 调用
# ───────────────────────────────────────────────
def call_qwen(prompt: str, retries: int = 3) -> str:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.3,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""   # 失败返回空，后续用原始描述兜底
    return ""


# ───────────────────────────────────────────────
# 加载已有进度（支持断点续跑）
# ───────────────────────────────────────────────
def load_existing():
    done = {}
    if os.path.isfile(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[(row["type"], row["original_id"])] = row["rewritten"]
    return done


# ───────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────
def main():
    print("连接数据库...")
    conn = get_db_conn()
    items_by_type = {}
    for item_type, sql in QUERIES.items():
        rows = query_db(conn, sql)
        items_by_type[item_type] = rows
        print(f"  {item_type}: {len(rows)} 条")
    conn.close()

    # 加载已有进度
    done = load_existing()
    print(f"\n已完成：{len(done)} 条，继续未完成部分...")

    # 追加写入
    is_new_file = not os.path.isfile(OUTPUT_PATH) or len(done) == 0
    f_out = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_out)
    if is_new_file:
        writer.writerow(["type", "original_id", "rewritten"])

    total = sum(len(v) for v in items_by_type.values())
    pbar = tqdm(total=total, desc="改写进度")

    for item_type in ["shop", "school", "course", "apartment"]:
        prompt_tpl = PROMPTS[item_type]
        for row in items_by_type[item_type]:
            key = (item_type, str(row["id"]))
            pbar.update(1)
            if key in done:
                continue

            # 拼接原始文本
            name  = str(row.get("name") or "").strip()
            desc  = str(row.get("description") or "").strip()
            addr  = str(row.get("address") or "").strip()
            original = "。".join(p for p in [name, desc, addr] if p)

            if not original.strip():
                rewritten = ""
            else:
                prompt = prompt_tpl.format(name=name, description=desc, address=addr)
                rewritten = call_qwen(prompt)
                time.sleep(REQUEST_INTERVAL)

            writer.writerow([item_type, row["id"], rewritten])
            f_out.flush()

    pbar.close()
    f_out.close()
    print(f"\n完成！结果保存至 {OUTPUT_PATH}")
    print("下一步：在 prepare_features.py 的 build_text() 中读取 rewritten 列作为文本输入。")


if __name__ == "__main__":
    main()
