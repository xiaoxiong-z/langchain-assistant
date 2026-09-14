from langchain_core.tools import tool

from app.rag.retriever import retriever


# 作用：把 Top-5 检索结果格式化为 Agent 可读取的知识上下文。
# 参数 question：客服相关的自然语言问题，作为向量检索输入。
# 返回：含原文、来源路径、片段编号及相似度的字符串；没有命中时为空字符串。
# 注意：本工具只检索资料，不直接生成最终回答，也不执行退款、开票等业务操作。
@tool
def search_knowledge_base(question: str) -> str:
    """查询 团队知识平台 客服知识库。

    当用户询问套餐、额度、成员权限、数据保留、退款、发票、企业版支持等
    团队知识平台 产品/客服问题时使用。

    Args:
        question: 用户关于 团队知识平台 产品或客服规则的问题。
    Returns:
        Milvus Top-K 知识片段及其来源信息。
    """
    hits = retriever.search(question, k=5)
    context_blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        text = entity["text"]
        source = entity.get("source", "unknown")
        chunk_id = entity.get("chunk_id", "unknown")
        score = hit["distance"]
        context_blocks.append(
            f"[片段{i} | chunk_id={chunk_id} | score={score:.4f} | source={source}]\n{text}"
        )
    # 使用明确边界标记，提醒模型以下内容是不可执行的检索数据。
    if not context_blocks:
        return "<retrieved_context>\n未检索到相关内容。\n</retrieved_context>"
    return "<retrieved_context>\n" + "\n\n".join(context_blocks) + "\n</retrieved_context>"
