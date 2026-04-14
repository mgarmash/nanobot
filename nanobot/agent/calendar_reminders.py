from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage


def _extract_link(event: dict[str, Any]) -> str | None:
    url = event.get("url")
    if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
        return url

    location = str(event.get("location") or "").strip()
    if location.startswith("http://") or location.startswith("https://"):
        return location

    description = str(event.get("description") or "")
    import re

    match = re.search(r"https?://\S+", description)
    if match:
        return match.group(0).rstrip(").,;")
    return None


def _hhmm(iso_value: str, timezone_name: str) -> str:
    from zoneinfo import ZoneInfo

    dt = datetime.fromisoformat(iso_value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(ZoneInfo(timezone_name))
    return dt.strftime("%H:%M")


def _dedupe_key(event: dict[str, Any], lead_minutes: int) -> str:
    return f"{event.get('id')}|{event.get('starts_at')}|{lead_minutes}"


class CalendarReminderState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, bool]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            logger.warning("Failed to load calendar reminder state")
        return {}

    def save(self, sent: dict[str, bool]) -> None:
        self.path.write_text(json.dumps(sent, ensure_ascii=False, indent=2), encoding="utf-8")


class CalendarReminderRunner:
    def __init__(
        self,
        *,
        tool_executor,
        outbound_sender,
        state_path: Path,
        channel: str,
        chat_id: str,
        message_thread_id: str,
        timezone: str,
        lead_minutes: int = 5,
        interval_seconds: int = 300,
    ):
        self._tool_executor = tool_executor
        self._outbound_sender = outbound_sender
        self._state = CalendarReminderState(state_path)
        self._channel = channel
        self._chat_id = chat_id
        self._message_thread_id = message_thread_id
        self._timezone = timezone
        self._lead_minutes = lead_minutes
        self._interval_seconds = interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Calendar reminder loop failed: {}", exc)
            await asyncio.sleep(self._interval_seconds)

    async def run_once(self) -> None:
        raw = await self._tool_executor(
            "mcp_calendar_events_starting_soon",
            {"lead_minutes": self._lead_minutes},
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return
        events = data.get("events", [])
        if not isinstance(events, list):
            return

        sent = self._state.load()
        changed = False

        for event in events:
            if not isinstance(event, dict):
                continue
            key = _dedupe_key(event, self._lead_minutes)
            if sent.get(key):
                continue

            title = str(event.get("title") or "Событие")
            calendar_kind = str(event.get("calendar_kind") or "?")
            starts_at = str(event.get("starts_at") or "")
            ends_at = str(event.get("ends_at") or "")
            time_range = (
                f"{_hhmm(starts_at, self._timezone)}–{_hhmm(ends_at, self._timezone)}"
                if starts_at and ends_at
                else ""
            )
            link = _extract_link(event)

            lines = [f"Через {self._lead_minutes} минут:", "", f"• {title} [{calendar_kind}]"]
            if time_range:
                lines.append(f"• {time_range}")
            if link:
                lines.append(f"• {link}")

            await self._outbound_sender(
                OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content="\n".join(lines),
                    metadata={"message_thread_id": self._message_thread_id},
                )
            )
            logger.info(
                "Calendar reminder sent: {} [{}] at {} to {}:{} thread {}",
                title,
                calendar_kind,
                starts_at,
                self._channel,
                self._chat_id,
                self._message_thread_id,
            )
            sent[key] = True
            changed = True

        if changed:
            self._state.save(sent)
