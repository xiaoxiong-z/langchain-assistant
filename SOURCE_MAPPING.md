# 课程代码映射

## 第4章：多轮对话聊天机器人

来源位置：
- 1.6.2：`keep_recent_messages()` 窗口记忆函数（PDF 第17页）
- 1.6.3：多轮对话聊天机器人（PDF 第18–19页）

本项目对应：
- `memory.py`
- `assistant.py` 中的 `self.messages`、`chat()`、`reset()`

保留内容：消息列表、多轮历史、最近 N 轮窗口、退出/重置式交互思路。

## 第7章：多功能智能助手

来源位置：
- 第9节“实战：多功能智能助手”（PDF 第76–82页）

本项目对应：
- `business_tools.py`
- `assistant.py`
- `main.py`

保留的 5 个工具：
- `get_weather`
- `calculator`
- `get_time_info`
- `convert_currency`
- `search_info`

保留 Agent 核心：`create_agent()`、系统提示词、工具列表、交互主循环。

## 第10章：Atguigu Assistant 客服知识库

来源位置：
- 2.5.3 案例：Atguigu Assistant 客服知识库（PDF 第52–65页）

本项目对应：
- `rag_tool.py`
- `knowledge.txt`

保留 RAG 链路：
- Milvus 初始化
- BGE-M3 / 1024 维 Embedding
- `TextLoader`
- `RecursiveCharacterTextSplitter`
- `chunk_size=220`
- `chunk_overlap=80`
- 文档向量化与 Milvus upsert
- COSINE Top-K 检索

整合改造：
- 原案例的 `retrieve()` 被继续保留；
- 新增 `search_knowledge_base` Tool，仅用于把检索能力接入第7章 Agent；
- 第10章“仅根据检索上下文回答；上下文不足回答不知道”的约束合并进统一 Agent 的 system prompt。
