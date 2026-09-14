"""可选的命令行入口，用于验证核心 Agent 与 FastAPI 解耦。"""

from app.agent.assistant import AssistantService
from app.rag.retriever import retriever


# 作用：启动命令行交互循环，处理普通问题、reset 和 quit。
# 参数：无；返回：None；通过标准输入读取问题，向终端打印结果。
# 启动时连接已有知识库；历史由本函数保存，成功回答后再追加。
def main():
    retriever.initialize()
    assistant = AssistantService()
    history: list[dict[str, str]] = []

    print("小谷姐姐 CLI，输入 quit 退出，reset 清空历史。")
    while True:
        question = input("\n👤 你: ").strip()
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            history.clear()
            print("✅ 已重置")
            continue
        if not question:
            continue

        answer, tools_used = assistant.chat(history, question)
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        print(f"🤖 小谷姐姐: {answer}")
        if tools_used:
            print(f"🔧 tools: {', '.join(tools_used)}")


if __name__ == "__main__":
    main()
