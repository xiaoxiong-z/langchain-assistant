from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock, RLock
from uuid import uuid4


@dataclass
class SessionState:
    # 会话唯一标识，用于关联后续请求与对应的历史记录。
    session_id: str
    # 当前会话的消息列表；default_factory 保证各会话独立创建列表。
    messages: list[dict[str, str]] = field(default_factory=list)
    # 创建时间（会话管理器使用 UTC）。
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 最近一次成功保存聊天结果的 UTC 时间。
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 每个会话独立的锁，用来串行处理同一会话的聊天请求。
    lock: Lock = field(default_factory=Lock, repr=False)

    # 作用：把消息总数除以 2，得到已保存的完整对话轮数。
    # 参数：仅 self；返回：整数。假定消息按 user/assistant 成对保存。
    @property
    def history_pairs(self) -> int:
        return len(self.messages) // 2


class SessionManager:
    """内存版多会话管理器。

    这是 FastAPI 工程化扩展：用 session_id 隔离不同用户的聊天历史。
    进程重启后会话会丢失，适合学习 / MVP；生产环境可替换为 Redis/数据库。
    """

    # 作用：创建进程内会话字典与可重入锁；参数：仅 self。
    # 返回：无业务返回值；锁保护字典操作，多进程之间不共享这些会话。
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._lock = RLock()

    # 作用：生成 UUID 会话 ID，并把新的 SessionState 保存到内存字典。
    # 参数：仅 self；返回：新建会话对象。字典写入受管理器锁保护。
    def create(self) -> SessionState:
        session = SessionState(session_id=uuid4().hex)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    # 作用：按 ID 读取会话对象，不创建新会话。
    # 参数 session_id：会话唯一标识；返回：SessionState，未找到则返回 None。
    # 返回的是共享对象；聊天历史的并发读写还需使用该会话自身的锁。
    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    # 作用：按 ID 从内存字典移除会话。
    # 参数 session_id：目标会话 ID；返回：删除成功为 True，不存在为 False。
    # 已取得该对象引用的在途请求不会因此自动取消。
    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    # 作用：统计当前进程保存的会话数；参数：仅 self；返回：整数。
    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
