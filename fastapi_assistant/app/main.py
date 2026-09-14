from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import API_TITLE, API_VERSION, CORS_ORIGINS
from app.rag.retriever import retriever
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


# 作用：管理 FastAPI 启动阶段，连接已有知识库；失败时允许服务降级启动。
# 参数 app：FastAPI 传入的应用实例（此函数体未直接使用）。
# yield 前执行初始化，yield 处把控制权交回框架；当前没有额外的关闭清理逻辑。
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 这里只连接已有知识库，不进行文档切分、Embedding 批量建库。
    # 建库由 scripts/build_knowledge_base.py 独立负责。
    try:
        retriever.initialize()
        print("✅ RAG 检索器初始化完成")
    except Exception as exc:
        # 服务仍然启动，/health 显示 degraded；非 RAG 工具仍可用于学习测试。
        print(f"⚠️ RAG 检索器未就绪：{exc}")
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "将多轮对话、多功能 Agent、 团队知识平台 RAG "
        "整合为可通过 HTTP 调用的 FastAPI 后端。"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web"), name="static")


# 作用：提供服务首页信息；参数：无。
# 返回：含服务名称、版本、接口文档和健康检查地址的字典。
@app.get("/", tags=["system"])
def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }

@app.get("/app", include_in_schema=False)
def web_app():
    return FileResponse(Path(__file__).parent / "web" / "index.html")
