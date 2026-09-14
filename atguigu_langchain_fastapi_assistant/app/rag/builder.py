from __future__ import annotations

from langchain.embeddings import init_embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

from app.core.config import (
    COLLECTION_NAME,
    DB_NAME,
    EMBED_DIM,
    EMBED_MODEL_NAME,
    KNOWLEDGE_FILE,
    MILVUS_URI,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)


# 作用：离线重建知识库；在线 API 启动流程不会调用此函数。
# 参数：无，数据库连接、文件和模型参数均来自配置模块。
# 返回：成功写入的文档片段数量（整数）。
# 副作用：删除同名集合并重新写入；文档发送到 Embedding 服务。
# 异常：文件不存在时抛出 FileNotFoundError；连接、向量化或写入失败向上抛出。
def build_knowledge_base() -> int:
    """离线构建知识库。

    保留第10章案例的核心链路：
    TextLoader -> RecursiveCharacterTextSplitter -> Embedding -> Milvus upsert。
    """
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"知识库文件不存在：{KNOWLEDGE_FILE}")

    client = MilvusClient(MILVUS_URI)
    if DB_NAME not in client.list_databases():
        client.create_database(db_name=DB_NAME)
    client.use_database(db_name=DB_NAME)

    # 建库脚本显式重建 collection；在线 API 不再执行这一步。
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBED_DIM,
        # 使用余弦相似度衡量方向接近程度；这里返回的分数越高通常越相似。
        metric_type="COSINE",
    )

    embed_model = init_embeddings(
        model="openai:" + EMBED_MODEL_NAME,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
    )

    documents = TextLoader(str(KNOWLEDGE_FILE), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(
        # 每块目标上限为 220 个字符（默认按字符计数，不是 token 数）。
        chunk_size=220,
        # 相邻片段最多重叠 80 个字符，尽量保留跨片段语义。
        chunk_overlap=80,
        # 按顺序尝试分隔符：优先章节/段落，再到标点，空字符串兜底按字符拆分。
        separators=[
            "\n==============================\n",
            "\n\n",
            "\n",
            "。",
            "，",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents(documents)

    vectors = embed_model.embed_documents([chunk.page_content for chunk in chunks])
    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": chunks[i].page_content,
            "source": str(KNOWLEDGE_FILE),
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)
    return len(chunks)
