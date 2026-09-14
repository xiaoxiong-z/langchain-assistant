from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    # 会话唯一标识，用于关联后续请求与对应的历史记录。
    session_id: str
    # 创建时间（会话管理器使用 UTC）。
    created_at: datetime


class ChatRequest(BaseModel):
    # 会话唯一标识，用于关联后续请求与对应的历史记录。
    session_id: str = Field(min_length=8, description="由 POST /sessions 创建的会话 ID")
    # 本轮输入文本，长度为 1～4000 字符；当前没有额外剔除纯空白输入。
    message: str = Field(min_length=1, max_length=4000)

class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=200)

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    session_id: str


class ChatResponse(BaseModel):
    # 会话唯一标识，用于关联后续请求与对应的历史记录。
    session_id: str
    # Agent 最终返回给用户的文本。
    answer: str
    # 本轮 AI 消息中出现的工具名，按出现顺序去重。
    tools_used: list[str]
    # 当前会话累计保存的对话轮数，不是窗口截取后的轮数。
    history_pairs: int


class DeleteSessionResponse(BaseModel):
    # 会话唯一标识，用于关联后续请求与对应的历史记录。
    session_id: str
    # 是否成功从内存会话字典中删除目标会话。
    deleted: bool


class HealthResponse(BaseModel):
    # 整体状态：配置存在且检索器就绪为 ok，否则为 degraded。
    status: str
    # 当前进程中的会话数量，并非实时在线用户数。
    active_sessions: int
    # 是否同时设置聊天密钥和接口地址，不验证密钥有效性。
    llm_configured: bool
    # 是否同时设置向量服务密钥和接口地址。
    embedding_configured: bool
    # 检索器最近一次初始化是否成功，不是实时连通性检查。
    rag_ready: bool
    # 检索器记录的初始化错误；无错误时为 None。
    rag_error: str | None = None
