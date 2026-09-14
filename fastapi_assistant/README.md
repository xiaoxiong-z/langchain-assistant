# Knowledge Assistant API

基于 FastAPI 的知识库问答后端，将 LangChain Agent、Milvus 检索与多轮会话管理封装为 HTTP 服务。

## 启动

```powershell
conda activate agent
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# 首次建库或更新文档时执行，会重建同名集合
python -m scripts.build_knowledge_base
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

启动后访问 `http://127.0.0.1:8000/app`，可直接注册、登录并聊天；接口调试页仍在 `/docs`。用户密码保存在本地 `users.db`，登录令牌和会话目前保存在进程内存中，适合本地演示。

聊天模型需支持工具调用，文档与查询应使用相同 Embedding 模型及维度。已有匹配的知识库时可跳过建库。

## 接口

访问 `http://127.0.0.1:8000/docs`。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/api/v1/health` | 配置、检索器状态及会话数量 |
| POST | `/api/v1/sessions` | 创建会话 |
| POST | `/api/v1/chat` | 对话 |
| DELETE | `/api/v1/sessions/{session_id}` | 删除会话 |

先创建会话，再把返回的 ID 填入聊天请求：

```json
{
  "session_id": "替换为创建接口返回的ID",
  "message": "请查询知识库中的成员数量上限"
}
```

响应中 `answer` 为回答，`tools_used` 为工具调用名称，`history_pairs` 为累计保存的轮数。同一 ID 支持连续追问。

## 模块职责

- `app/api`：接口路由及请求/响应校验。
- `app/agent`：模型初始化、工具注册及回答提取。
- `app/memory`：进程内会话、同会话锁和消息窗口。
- `app/rag`：离线建库与在线检索。
- `app/tools`：知识检索与业务工具。
- `app/core`：配置管理。
- `scripts`：建库脚本及可选命令行入口。

## 使用边界

登录账号保存在本地 SQLite，令牌和会话目前存在进程内存中，自动重载会清空登录状态；天气、汇率与产品/新闻搜索为模拟数据；当前没有请求限流，生产环境应使用 HTTPS、持久化令牌和 Redis/数据库。
