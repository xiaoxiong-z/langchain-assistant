# Knowledge Assistant

基于 Python、LangChain 和 Milvus 的知识库智能助手，将多轮对话、语义检索和工具调用整合在同一条问答流程中。提供命令行入口与 FastAPI 后端，可用于客服知识查询及问答应用的后端接入。

## 核心功能

- RAG 知识库问答：读取 TXT 文档，递归切分并使用 BGE-M3 生成向量；基于 Milvus 召回 Top-5 片段。
- Agent 工具调用：接入知识库检索、计算、时间、天气、货币换算和信息搜索六类工具。
- 多轮对话：保存用户问题与最终回答，通过消息窗口控制上下文。
- HTTP 接口：提供聊天、会话创建与删除、健康检查接口，包含 Pydantic 校验和自动接口文档。
- 会话隔离：通过 session_id 管理独立历史，同一会话使用锁串行处理请求。
- 独立建库：FastAPI 版本将离线建库和在线检索分离，服务重启无需重复生成文档向量。

## 技术栈

Python 3.11 · LangChain · FastAPI · Pydantic · Milvus · BGE-M3 · OpenAI 兼容 API

## 项目结构

```text
.
├── main.py                 # 命令行入口
├── assistant.py            # Agent 与会话历史
├── business_tools.py       # 业务工具
├── rag_tool.py             # 知识库构建与检索工具
├── memory.py               # 历史窗口
├── config.py               # 命令行配置
├── knowledge.txt           # 示例知识库
├── .env.example            # 配置模板
└── fastapi_assistant/      # HTTP 服务版本
```

## 启动 FastAPI 版本

```powershell
conda activate agent
cd fastapi_assistant
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

填写 `.env` 中的聊天模型、Embedding 和 Milvus 配置。聊天模型需支持工具调用。

```powershell
# 首次使用或文档更新后执行，会重建同名集合
python -m scripts.build_knowledge_base
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/docs`，先创建会话，再使用返回的 ID 调用聊天接口。详见 [FastAPI 使用说明](fastapi_assistant/README.md)。

## 启动命令行版本

```powershell
conda activate agent
python -m pip install -r requirements.txt
python -X utf8 main.py
```

输入 `reset` 清空会话，输入 `quit` 退出。命令行版本每次启动都会重建知识库集合。

## 当前实现边界

- 天气、汇率及产品/新闻搜索使用预设数据，不是实时第三方服务。
- 会话保存在进程内存中，重启丢失，不支持多进程共享。
- 计算工具使用受限 eval，后端尚未实现身份认证和请求限流。
- 示例知识库中的套餐、权限与退款规则属于虚构产品资料。
- 密钥只保存在本地 `.env`，不要提交到版本库。
