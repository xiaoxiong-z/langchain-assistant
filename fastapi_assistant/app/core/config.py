import os
from pathlib import Path

from dotenv import load_dotenv

# 读取 .env；override=True 表示同名环境变量可被文件中的值覆盖。
load_dotenv(override=True)

# FastAPI 子项目根目录，从当前配置文件位置向上定位。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 默认知识库数据目录。
DATA_DIR = PROJECT_ROOT / "data"

# ===== 多轮对话窗口 =====
# 发送给模型的历史窗口参数，按每轮两条普通消息计算；建议为正整数。
MAX_PAIRS_HISTORY = int(os.getenv("MAX_PAIRS_HISTORY", "10"))
SUMMARY_TRIGGER_PAIRS = int(os.getenv("SUMMARY_TRIGGER_PAIRS", "12"))
SUMMARY_KEEP_PAIRS = int(os.getenv("SUMMARY_KEEP_PAIRS", "4"))

# ===== 聊天模型 =====
# 聊天模型 ID；由服务商识别，与 Embedding 模型独立配置。
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")
# 聊天服务密钥；保留历史变量名，也可配置 OpenRouter 等兼容服务。
CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
# 聊天服务 API 基础地址，需与密钥所属服务匹配。
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")

# ===== 团队知识平台 RAG =====
# Milvus 服务连接地址；该参数不负责启动数据库。
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# Milvus 数据库名称，用于隔离不同知识库集合。
DB_NAME = os.getenv("MILVUS_DB_NAME", "rag_tutorial")
# 存储文档向量的集合名；建库流程会重建同名集合。
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "docs")
# 待加载的 UTF-8 文本文件路径。
KNOWLEDGE_FILE = Path(os.getenv("KNOWLEDGE_FILE", str(DATA_DIR / "knowledge.txt")))
# 文档与查询问题共用的 Embedding 模型 ID，保证向量空间一致。
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "Pro/BAAI/bge-m3")
# 集合向量维度，必须与 Embedding 实际输出维度相同。
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
# 硅基流动 Embedding 服务密钥，从环境变量读取。
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
# Embedding API 基础地址，与聊天模型接口分开配置。
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")

# ===== FastAPI 工程扩展 =====
# 接口文档和首页中显示的服务名称。
API_TITLE = "Knowledge Assistant API"
# 接口文档展示的版本号。
API_VERSION = "2.0.0"
# 允许跨域请求的前端来源，按逗号拆分并去掉空白；这不是身份认证。
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if item.strip()
]
