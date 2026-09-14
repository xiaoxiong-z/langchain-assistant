from langchain.embeddings import init_embeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

from config import (
    COLLECTION_NAME,
    DB_NAME,
    EMBED_DIM,
    EMBED_MODEL_NAME,
    KNOWLEDGE_FILE,
    MILVUS_URI,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)

client = None
embed_model = None


# 作用：连接 Milvus、重建集合、读取 TXT、切分、向量化并写入知识库。
# 参数：无显式参数，使用 config 中的数据库、文件路径和模型配置。
# 返回：None；同时设置模块级 client 与 embed_model，供后续检索复用。
# 副作用：删除同名集合中的原有数据；文档文本会发送到 Embedding 服务。
def init_knowledge_base() -> None:
    """按知识库配置初始化并写入 Milvus。"""
    global client, embed_model

    # 1. 初始化 Milvus
    client = MilvusClient(MILVUS_URI)
    existing_dbs = client.list_databases()
    if DB_NAME not in existing_dbs:
        client.create_database(db_name=DB_NAME)
    client.use_database(db_name=DB_NAME)

    # 当前实现每次运行会重建 collection，便于学习时保证数据一致。
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBED_DIM,
        # 使用余弦相似度衡量方向接近程度；这里返回的分数越高通常越相似。
        metric_type="COSINE",
    )

    # 2. 初始化 Embedding
    embed_model = init_embeddings(
        model="openai:" + EMBED_MODEL_NAME,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
    )

    # 3. 文档加载与切分
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    documents = loader.load()
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

    # 4. 向量化并写入 Milvus
    vectors = embed_model.embed_documents([chunk.page_content for chunk in chunks])
    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": chunks[i].page_content,
            "source": KNOWLEDGE_FILE,
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)
    print(f"✅ 知识库初始化完成，共写入 {len(chunks)} 个 chunk")


# 作用：向量化问题，再从 Milvus 召回最相似的文档片段。
# 参数 question：自然语言问题；k：最多返回的片段数，默认 5，应为正整数。
# 返回：当前问题对应的命中列表；每项包含 entity 文档字段与 distance 相似度。
# 异常：未初始化知识库时抛出 RuntimeError；网络/检索错误由调用方处理。
def retrieve(question: str, k: int = 5):
    """通过向量相似度从 Milvus 召回最相关的 K 个文本片段。"""
    if client is None or embed_model is None:
        raise RuntimeError("知识库尚未初始化，请先调用 init_knowledge_base()")

    query_vector = embed_model.embed_query(question)
    results = client.search(
        collection_name=COLLECTION_NAME,
        # Milvus 接收一批查询向量；这里仅查询一个问题，所以外层列表长度为 1。
        data=[query_vector],
        # 每个查询最多召回 k 条结果，实际条数可能更少。
        limit=k,
        # 除命中 ID/分数外，额外返回原文、来源和片段编号。
        output_fields=["text", "source", "chunk_id"],
    )
    # 取批量查询中第一个问题的命中列表，而非只取一个文档。
    return results[0]


# 作用：把 Top-5 检索结果格式化为 Agent 可读取的知识上下文。
# 参数 question：客服相关的自然语言问题，作为向量检索输入。
# 返回：含原文、来源路径、片段编号及相似度的字符串；没有命中时为空字符串。
# 注意：本工具只检索资料，不直接生成最终回答，也不执行退款、开票等业务操作。
@tool
def search_knowledge_base(question: str) -> str:
    """查询 团队知识平台 客服知识库。

    当用户询问 团队知识平台 的套餐、额度、成员权限、数据保留、退款、发票、
    企业版支持等产品/客服问题时使用本工具。

    Args:
        question: 用户关于 团队知识平台 产品或客服规则的问题。
    Returns:
        Milvus 检索到的 Top-K 知识片段及其来源信息。
    """
    hits = retrieve(question, k=5)
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        text = hit["entity"]["text"]
        source = hit["entity"].get("source", "unknown")
        chunk_id = hit["entity"].get("chunk_id", "unknown")
        score = hit["distance"]
        context_blocks.append(
            f"[片段{i} | chunk_id={chunk_id} | score={score:.4f} | source={source}]\n{text}"
        )
    return "\n\n".join(context_blocks)
