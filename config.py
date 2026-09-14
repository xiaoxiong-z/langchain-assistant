import os
from pathlib import Path

from dotenv import load_dotenv

# 读取 .env；override=True 表示同名环境变量可被文件中的值覆盖。
load_dotenv(override=True)

# 当前命令行项目目录；定位知识库时不依赖启动命令所在目录。
BASE_DIR = Path(__file__).resolve().parent

# 多轮对话配置
# 聊天模型 ID；由服务商识别，与 Embedding 模型独立配置。
MODEL_NAME = "openrouter/free"
# 发送给模型的历史窗口参数，按每轮两条普通消息计算；建议为正整数。
MAX_PAIRS_HISTORY = 10
# 命令行退出口令，输入时不区分大小写。
EXIT_WORD = "quit"
# 命令行清空当前对话历史的口令。
RESET_WORD = "reset"

#  团队知识平台 RAG 案例配置
# Milvus 服务连接地址；该参数不负责启动数据库。
MILVUS_URI = "http://localhost:19530"
# Milvus 数据库名称，用于隔离不同知识库集合。
DB_NAME = "rag_tutorial"
# 存储文档向量的集合名；建库流程会重建同名集合。
COLLECTION_NAME = "docs"
# 待加载的 UTF-8 文本文件路径。
KNOWLEDGE_FILE = str(BASE_DIR / "knowledge.txt")
# 文档与查询问题共用的 Embedding 模型 ID，保证向量空间一致。
EMBED_MODEL_NAME = "Pro/BAAI/bge-m3"
# 集合向量维度，必须与 Embedding 实际输出维度相同。
EMBED_DIM = 1024

# 聊天服务密钥；保留历史变量名，也可配置 OpenRouter 等兼容服务。
CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
# 聊天服务 API 基础地址，需与密钥所属服务匹配。
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")
# 硅基流动 Embedding 服务密钥，从环境变量读取。
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
# Embedding API 基础地址，与聊天模型接口分开配置。
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")

