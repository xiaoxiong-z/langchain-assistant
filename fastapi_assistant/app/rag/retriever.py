from __future__ import annotations

from langchain.embeddings import init_embeddings
from pymilvus import MilvusClient

from app.core.config import (
    COLLECTION_NAME,
    DB_NAME,
    EMBED_MODEL_NAME,
    MILVUS_URI,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)


class KnowledgeRetriever:
    """只负责在线检索，不负责重新建库。"""

    # 作用：建立尚未连接的检索器初始状态；参数：仅 self。
    # client/embed_model 为客户端，ready 表示初始化是否成功，last_error 保存错误。
    # 返回：无业务返回值；需要随后调用 initialize 才能检索。
    def __init__(self):
        self.client: MilvusClient | None = None
        self.embed_model = None
        self.ready = False
        self.last_error: str | None = None

    # 作用：检查已有数据库和集合，并准备查询向量化客户端，不重新建库。
    # 参数：仅 self；返回：None。成功后 ready=True，并清空 last_error。
    # 失败时清空客户端、记录错误、设置 ready=False，然后继续抛出异常。
    # 此处创建 Embedding 客户端，不执行实际向量化请求。
    def initialize(self) -> None:
        """连接已有 Milvus collection 并初始化 Embedding 模型。"""
        try:
            client = MilvusClient(MILVUS_URI)
            if DB_NAME not in client.list_databases():
                raise RuntimeError(
                    f"Milvus 数据库 {DB_NAME!r} 不存在，请先执行 python -m scripts.build_knowledge_base"
                )
            client.use_database(db_name=DB_NAME)
            if not client.has_collection(collection_name=COLLECTION_NAME):
                raise RuntimeError(
                    f"Milvus collection {COLLECTION_NAME!r} 不存在，请先执行 python -m scripts.build_knowledge_base"
                )

            embed_model = init_embeddings(
                model="openai:" + EMBED_MODEL_NAME,
                api_key=SILICONFLOW_API_KEY,
                base_url=SILICONFLOW_BASE_URL,
            )

            self.client = client
            self.embed_model = embed_model
            self.ready = True
            self.last_error = None
        except Exception as exc:
            self.client = None
            self.embed_model = None
            self.ready = False
            self.last_error = str(exc)
            raise

    # 作用：把问题转为向量，在已初始化的集合中检索相关片段。
    # 参数 question：自然语言问题；k：最多召回的片段数量，默认 5，应为正整数。
    # 返回：命中字典列表，含 entity（原文和来源）及 distance（相似度）。
    # 检索器未就绪时抛出 RuntimeError；问题文本会发送到 Embedding 服务。
    def search(self, question: str, k: int = 5) -> list[dict]:
        if not self.ready or self.client is None or self.embed_model is None:
            raise RuntimeError(
                self.last_error
                or "RAG 检索器尚未初始化，请先构建知识库并检查 Milvus/Embedding 配置。"
            )

        query_vector = self.embed_model.embed_query(question)
        results = self.client.search(
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


retriever = KnowledgeRetriever()
