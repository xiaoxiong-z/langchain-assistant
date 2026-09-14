from app.rag.builder import build_knowledge_base


if __name__ == "__main__":
    print("开始构建 Atguigu Assistant 知识库...")
    count = build_knowledge_base()
    print(f"✅ 知识库构建完成，共写入 {count} 个 chunk")
