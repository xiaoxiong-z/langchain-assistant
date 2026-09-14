from assistant import SmartAssistant
from config import EXIT_WORD, RESET_WORD
from rag_tool import init_knowledge_base


# 作用：启动命令行交互循环，处理普通问题、reset 和 quit。
# 参数：无；返回：None；通过标准输入读取问题，向终端打印结果。
# 启动时重建知识库，再创建有状态助手。
def main():
    print("正在初始化 团队知识平台 知识库（Milvus + Embedding）...")
    init_knowledge_base()

    assistant = SmartAssistant()

    print("=" * 56)
    print("🤖 智能助手：多轮对话 + 多功能 Agent + RAG 客服知识库")
    print("=" * 56)
    print("我可以帮你：")
    print(" 🌤 查询天气")
    print(" 🔢 数学计算")
    print(" ⏰ 时间查询")
    print(" 💱 货币转换")
    print(" 🔍 产品 / 新闻模拟信息搜索")
    print(" 📚 团队知识平台 客服知识库问答")
    print(f"\n输入 '{EXIT_WORD}' 退出，输入 '{RESET_WORD}' 重置对话\n")

    while True:
        user_input = input("\n👤 你: ").strip()

        if user_input.lower() == EXIT_WORD:
            print("再见！👋")
            break

        if user_input.lower() == RESET_WORD:
            assistant.reset()
            print("✅ 对话已重置")
            continue

        if not user_input:
            continue

        response = assistant.chat(user_input)
        print(f"🤖 智能助手: {response}")


if __name__ == "__main__":
    main()
