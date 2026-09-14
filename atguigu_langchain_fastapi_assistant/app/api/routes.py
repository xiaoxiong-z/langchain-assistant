from datetime import datetime, timezone
from threading import Lock

from fastapi import APIRouter, HTTPException, status

from app.agent.assistant import AssistantService
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    DeleteSessionResponse,
    HealthResponse,
)
from app.core.config import (
    CLOSEAI_API_KEY,
    CLOSEAI_BASE_URL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)
from app.memory.session_manager import SessionManager
from app.rag.retriever import retriever

router = APIRouter()
session_manager = SessionManager()

_assistant_service: AssistantService | None = None
_assistant_init_lock = Lock()


# 作用：首次需要聊天时才创建 Agent，并复用同一服务实例。
# 参数：无；返回：AssistantService 实例。
# 初始化锁配合二次检查，避免多个线程同时重复创建；配置缺失时抛 RuntimeError。
def get_assistant_service() -> AssistantService:
    """延迟初始化模型/Agent，让服务即使配置不完整也能启动并查看 /health。"""
    global _assistant_service
    if _assistant_service is None:
        with _assistant_init_lock:
            if _assistant_service is None:
                if not CLOSEAI_API_KEY or not CLOSEAI_BASE_URL:
                    raise RuntimeError("聊天模型未配置：请检查 CLOSEAI_API_KEY / CLOSEAI_BASE_URL")
                _assistant_service = AssistantService()
    return _assistant_service


# 作用：返回模型配置是否存在、检索器初始化状态和内存会话数量。
# 参数：无；返回：HealthResponse，状态为 ok 或 degraded。
# 注意：此处不实时请求模型或数据库，配置存在也不代表密钥有效。
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    llm_configured = bool(CLOSEAI_API_KEY and CLOSEAI_BASE_URL)
    embedding_configured = bool(SILICONFLOW_API_KEY and SILICONFLOW_BASE_URL)
    all_ready = llm_configured and embedding_configured and retriever.ready

    return HealthResponse(
        status="ok" if all_ready else "degraded",
        active_sessions=session_manager.count(),
        llm_configured=llm_configured,
        embedding_configured=embedding_configured,
        rag_ready=retriever.ready,
        rag_error=retriever.last_error,
    )


# 作用：创建独立的内存会话，供后续聊天请求通过 ID 访问。
# 参数：无；返回：CreateSessionResponse，包含会话 ID 与 UTC 创建时间。
@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["session"],
)
def create_session() -> CreateSessionResponse:
    session = session_manager.create()
    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
    )


# 作用：从会话管理器中移除指定会话，不删除知识库或其他会话。
# 参数 session_id：URL 路径中的会话 ID。
# 返回：DeleteSessionResponse；会话不存在时返回 HTTP 404。
@router.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    tags=["session"],
)
def delete_session(session_id: str) -> DeleteSessionResponse:
    deleted = session_manager.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return DeleteSessionResponse(session_id=session_id, deleted=True)


# 作用：处理一次 HTTP 聊天请求，查找会话、调用 Agent，成功后保存一轮消息。
# 参数 request：经 Pydantic 校验的 ChatRequest，含 session_id 和 message。
# 返回：ChatResponse，包含回答、工具名和累计历史轮数。
# 错误状态：会话缺失为 404，RuntimeError 为 503，其他调用异常为 500。
@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    session = session_manager.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在，请先调用 POST /sessions")

    # 同一个 session 的并发请求串行化，避免历史消息交叉写入。
    # 不同会话可独立处理；同一会话从读历史到写结果保持串行。
    with session.lock:
        try:
            assistant_service = get_assistant_service()
            answer, tools_used = assistant_service.chat(
                history=session.messages,
                user_input=request.message,
            )
        except RuntimeError as exc:
            # 例如：聊天模型未配置，或用户触发 RAG Tool 但知识库尚未就绪。
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent 调用失败：{exc}") from exc

        session.messages.append({"role": "user", "content": request.message})
        session.messages.append({"role": "assistant", "content": answer})
        session.updated_at = datetime.now(timezone.utc)

        return ChatResponse(
            session_id=session.session_id,
            answer=answer,
            tools_used=tools_used,
            history_pairs=session.history_pairs,
        )
