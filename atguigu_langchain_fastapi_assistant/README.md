# Atguigu Enterprise AI Assistant — FastAPI 版

这是在上一版“第4章 + 第7章 + 第10章”融合项目上继续工程化后的版本。

核心原则是：**LangChain 负责 AI 能力，FastAPI 负责把 AI 能力暴露为可被网页、App 或其他系统调用的 HTTP 服务。**

## 1. 项目能力

课程原始能力：

- 第4章：多轮消息历史 + `keep_recent_messages()` 最近 N 轮窗口。
- 第7章：`create_agent()` + 天气、计算、时间、货币转换、模拟信息搜索 5 个 Tool。
- 第10章：TextLoader → RecursiveCharacterTextSplitter → Embedding → Milvus → Top-K Retrieval，并将检索封装为 `search_knowledge_base` Tool。

本版本新增的工程化能力：

- FastAPI REST API。
- Pydantic 请求/响应 Schema。
- `session_id` 多会话隔离。
- 同一 session 并发请求锁，避免消息历史交叉写入。
- RAG “离线建库”和“在线检索”彻底拆开。
- `/health` 健康检查；RAG 未就绪时服务仍可启动并显示 `degraded`。
- CORS 配置，方便后续接 Vue / React。
- 响应返回 `tools_used`，方便观察 Agent 本轮实际调用了哪些 Tool。

> FastAPI、SessionManager、CORS、HTTP 接口属于本项目的工程化扩展，不是课程第4/7/10章原始代码。

## 2. 架构

```text
Browser / Vue / React / Postman
             │ HTTP
             ▼
          FastAPI
             │
      session_id 隔离
             │
             ▼
      LangChain Agent
      ┌──────┼─────────────┐
      ▼      ▼             ▼
  Business  Memory       RAG Tool
   Tools    Window          │
                            ▼
                        Embedding
                            │
                            ▼
                          Milvus
```

## 3. 目录

```text
atguigu_langchain_fastapi_assistant/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── agent/
│   │   └── assistant.py
│   ├── core/
│   │   └── config.py
│   ├── memory/
│   │   ├── session_manager.py
│   │   └── window.py
│   ├── rag/
│   │   ├── builder.py
│   │   └── retriever.py
│   └── tools/
│       ├── business_tools.py
│       └── knowledge_tool.py
├── scripts/
│   ├── build_knowledge_base.py
│   └── cli.py
├── data/
│   └── knowledge.txt
├── .env.example
├── requirements.txt
├── SOURCE_MAPPING.md
└── README.md
```

## 4. 为什么拆分 RAG 建库与在线查询

旧版启动时会：读取文档 → 切分 → 批量 Embedding → 删除/重建 Milvus collection → 写入向量。

这适合课程演示，但不适合 Web API。否则每次服务重启都会重新生成整个知识库。

现在改为：

```text
离线：python -m scripts.build_knowledge_base
knowledge.txt → split → embed_documents → Milvus upsert

在线：FastAPI /chat
question → Agent → search_knowledge_base → embed_query → Milvus search
```

## 5. 启动

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 配置环境变量

```bash
cp .env.example .env
```

填写聊天模型、Embedding API 配置，并确保 Milvus 在默认地址运行：

```text
http://localhost:19530
```

### 5.3 首次构建知识库

```bash
python -m scripts.build_knowledge_base
```

只有知识库内容发生变化时才需要重新执行。

### 5.4 启动 FastAPI

```bash
uvicorn app.main:app --reload
```

然后访问：

```text
http://127.0.0.1:8000/docs
```

FastAPI 会自动生成 Swagger API 页面。

## 6. API 使用流程

### 6.1 健康检查

```http
GET /api/v1/health
```

正常示例：

```json
{
  "status": "ok",
  "active_sessions": 0,
  "rag_ready": true,
  "rag_error": null
}
```

如果没有先建库，服务仍然启动，但 `status` 会是 `degraded`。

### 6.2 创建会话

```http
POST /api/v1/sessions
```

返回：

```json
{
  "session_id": "...",
  "created_at": "..."
}
```

### 6.3 对话

```http
POST /api/v1/chat
Content-Type: application/json
```

```json
{
  "session_id": "上一步得到的 session_id",
  "message": "专业版多少钱？"
}
```

响应示例：

```json
{
  "session_id": "...",
  "answer": "...",
  "tools_used": ["search_knowledge_base"],
  "history_pairs": 1
}
```

继续在同一个 `session_id` 下提问，就能获得多轮上下文。

### 6.4 删除 / 重置会话

```http
DELETE /api/v1/sessions/{session_id}
```

删除后原会话历史不再存在。

## 7. 多用户为什么不会串话

课程 Demo 通常只有一个 `messages` 列表。Web 服务如果所有人共用同一个列表，就会出现 A 用户的历史被 B 用户看到的问题。

本项目改为：

```text
session A -> messages A
session B -> messages B
session C -> messages C
```

`SessionManager` 只保存每个会话的 user/final-assistant 历史。每次调用前继续复用第4章的 `keep_recent_messages()` 控制最近 N 轮窗口。

当前 Session 存在进程内存中，因此重启会丢失；这是学习/MVP设计。真实生产环境可替换为 Redis 或数据库。

## 8. 可选 CLI

如果你想验证“Agent 核心与 FastAPI 接口是解耦的”，可以：

```bash
python -m scripts.cli
```

## 9. 简历描述可升级为

> 基于 LangChain + FastAPI 构建企业知识库智能助手后端，整合 Agent、Function Calling、多轮会话与 Milvus RAG，通过 REST API 对外提供问答服务；设计基于 session_id 的多会话隔离机制，并将知识库离线构建与在线 Top-K 检索解耦，避免服务启动阶段重复向量化与索引重建。

## 10. 仍然属于学习版的部分

- 天气、汇率、产品/新闻工具仍是第7章课程中的模拟数据，并非真实第三方 API。
- `calculator` 仍沿用课程受限 `eval()` 示例，不建议直接用于生产环境。
- Session 使用进程内存，不支持多实例共享，也不持久化。
- 没有鉴权、限流、Redis、数据库、Docker、前端，这些可以作为下一阶段继续工程化的方向。
