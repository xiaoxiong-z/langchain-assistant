# Atguigu LangChain Integrated Assistant

这是把课程中的 3 个完整实战项目合成后的单一项目：

- 第4章：多轮对话聊天机器人
- 第7章：多功能智能助手
- 第10章：Atguigu Assistant 客服知识库

## 整合后的职责

- **第4章**：提供普通 user/assistant 消息历史与最近 N 轮窗口记忆。
- **第7章**：作为主框架，使用 `create_agent()` + Tools 决定调用天气、计算、时间、货币、模拟信息搜索等工具。
- **第10章**：保留 TextLoader → RecursiveCharacterTextSplitter → Embedding → Milvus → Top-K Retrieval 流程，并把检索能力封装成 `search_knowledge_base` Tool 接入第7章 Agent。

## 为什么 RAG 要改成 Tool

第10章原案例是“检索 → 拼接上下文 → 单独 Agent 生成回答”。合并后，为了让一个 Agent 同时拥有业务工具和知识库能力，本项目只做一个必要的整合改造：**将第10章 `retrieve()` 的结果封装为 `search_knowledge_base` 工具，并加入第7章的工具列表。**

这样用户可以在同一个多轮会话里混合提问，例如：

- “北京今天天气怎么样？” → `get_weather`
- “100 美元等于多少人民币？” → `convert_currency`
- “基础版支持多少成员？” → `search_knowledge_base`
- “先告诉我基础版价格，再帮我算 3 个用户一个月多少钱。” → 先 RAG，再 Calculator

## 关于第4章窗口记忆的整合方式

第4章的 `keep_recent_messages()` 默认把每轮看成 `user + assistant` 两条消息；但 Agent 工具调用过程中会产生额外的 tool-call / ToolMessage。

因此最终项目只持久化：

1. 用户原始问题
2. Agent 最终回答

每次新请求前，再对这两类普通对话消息应用最近 N 轮窗口。这样保留第4章的窗口记忆思路，同时不会从中间切断 Agent 的工具调用消息链。

## 目录

```text
atguigu_langchain_integrated_assistant/
├── main.py              # 运行入口
├── assistant.py         # Agent + 多轮会话整合
├── business_tools.py    # 第7章 5 个工具
├── rag_tool.py          # 第10章 RAG + Milvus + RAG Tool
├── memory.py            # 第4章 keep_recent_messages
├── config.py            # 统一配置
├── knowledge.txt        # 课件 Atguigu Assistant 知识库内容重建版
├── .env.example
├── requirements.txt
└── SOURCE_MAPPING.md
```

## 环境准备

1. Python 环境安装依赖：

```bash
pip install -r requirements.txt
```

2. 将 `.env.example` 复制为 `.env`，填写课件使用的模型与 Embedding API 配置。

3. 按第10章案例准备 Milvus，默认连接：

```text
http://localhost:19530
```

4. 运行：

```bash
python main.py
```

## 说明

- `knowledge.txt` 是依据已上传第10章课件中案例展示出的知识库内容重建的运行素材；原始课程资产 `knowledge.txt` 本身并未单独上传。
- 第7章“实战：多功能智能助手”本身使用的是 5 个 Tool + `create_agent` + 对话历史，并没有在这个实战段落中直接使用 Pydantic 结构化输出或 Middleware。因此本次合并没有为了“显得更复杂”而擅自加入它们。
- 第7章 Calculator 示例使用受限 `eval()`，这里按课件逻辑保留；它适合作为学习 Demo，不应直接当作生产环境的任意表达式执行方案。
- 第10章案例会在启动时删除并重建同名 Milvus collection，本项目也保留该学习版行为。
