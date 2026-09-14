# 课程来源与工程化扩展映射

## A. 第4章：多轮对话聊天机器人

课程来源：
- 1.6.2 `keep_recent_messages()`：保留最近 N 轮 user + assistant 对话。
- 1.6.3 多轮聊天机器人：维护消息列表并持续对话。

项目文件：
- `app/memory/window.py`
- `app/memory/session_manager.py`（Session 隔离属于工程扩展）
- `app/agent/assistant.py`

## B. 第7章：多功能智能助手

课程来源：第9节“实战：多功能智能助手”。

保留 Tool：
- `get_weather`
- `calculator`
- `get_time_info`
- `convert_currency`
- `search_info`

保留核心：
- `create_agent()`
- 系统提示词
- 工具自主选择
- 多工具顺序调用

项目文件：
- `app/tools/business_tools.py`
- `app/agent/assistant.py`

## C. 第10章：Atguigu Assistant 客服知识库

课程来源：2.5.3 Atguigu Assistant 客服知识库。

保留：
- `TextLoader`
- `RecursiveCharacterTextSplitter`
- `chunk_size=220`
- `chunk_overlap=80`
- BGE-M3 / 1024 维 Embedding
- Milvus / COSINE
- `embed_documents` 批量建库
- `embed_query` + Top-K 检索
- “仅基于检索上下文回答，上下文不足则不知道”的约束

项目文件：
- `app/rag/builder.py`
- `app/rag/retriever.py`
- `app/tools/knowledge_tool.py`
- `data/knowledge.txt`

整合改造：把第10章检索能力包装为 `search_knowledge_base` Tool，交给第7章 Agent 自主调用。

## D. 非课程原始代码：FastAPI 工程化扩展

以下内容是为了把课程 Demo 变成 HTTP 后端服务新增的，不应描述为课程原代码：

- `app/main.py`：FastAPI 应用、CORS、lifespan。
- `app/api/routes.py`：REST API。
- `app/api/schemas.py`：Pydantic 请求/响应模型。
- `app/memory/session_manager.py`：session_id 隔离和会话锁。
- `scripts/build_knowledge_base.py`：将第10章建库阶段拆成离线命令。
- `/health`：服务/RAG状态检查。
