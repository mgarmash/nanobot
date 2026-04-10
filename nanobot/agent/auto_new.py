"""Auto session new: proactive archival of idle sessions."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Coroutine

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.memory import Consolidator
    from nanobot.session.manager import Session, SessionManager


class AutoSessionNew:
    """Manages proactive archival of idle sessions.

    Monitors session idle time and archives expired sessions in the
    background so that the user experiences zero latency when returning.
    """

    def __init__(
        self,
        sessions: SessionManager,
        consolidator: Consolidator,
        session_ttl_minutes: int = 0,
    ):
        self.sessions = sessions
        self.consolidator = consolidator
        self._session_ttl_minutes = session_ttl_minutes
        self._archiving_keys: set[str] = set()
        self._pending_summaries: dict[str, str] = {}

    def is_expired(self, updated_at: datetime | str | None) -> bool:
        """Check whether an updated_at timestamp is beyond the TTL."""
        if self._session_ttl_minutes <= 0 or not updated_at:
            return False
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elapsed_s = (datetime.now() - updated_at).total_seconds()
        return elapsed_s >= self._session_ttl_minutes * 60

    async def check_expired(self, schedule_background: Callable[[Coroutine], None]) -> None:
        """Scan all sessions and schedule background archival for expired ones."""
        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            if not key or key in self._archiving_keys:
                continue
            if self.is_expired(info.get("updated_at")):
                self._archiving_keys.add(key)
                schedule_background(self._archive_and_store(key))

    async def _archive_and_store(self, session_key: str) -> None:
        """Archive an expired session in the background, store summary for next message."""
        try:
            summary = await self.archive_and_clear(session_key)
            if summary:
                self._pending_summaries[session_key] = summary
        except Exception:
            logger.exception("Proactive auto-new failed for {}", session_key)
        finally:
            self._archiving_keys.discard(session_key)

    async def archive_and_clear(self, session_key: str) -> str | None:
        """Archive un-consolidated messages and clear session.

        Returns the summary text (or None).
        """
        session = self.sessions.get_or_create(session_key)
        unconsolidated = session.messages[session.last_consolidated:]
        if not unconsolidated:
            return None

        logger.info("Auto session new for {} (idle {} min)", session_key, self._session_ttl_minutes)

        await self.consolidator.archive(unconsolidated)

        entries = self.consolidator.store.read_unprocessed_history(since_cursor=0)
        summary_text = entries[-1]["content"] if entries else ""
        if not summary_text or summary_text == "(nothing)":
            summary_text = ""

        session.clear()
        self.sessions.save(session)
        self.sessions.invalidate(session_key)

        return summary_text or None

    def needs_reload(self, session: Session, session_key: str) -> bool:
        """Check if session should be reloaded (archived or expired)."""
        return session_key in self._archiving_keys or self.is_expired(session.updated_at)

    def pop_summary(self, session_key: str) -> str | None:
        """Pop and return the pending summary for a session (one-shot)."""
        return self._pending_summaries.pop(session_key, None)
