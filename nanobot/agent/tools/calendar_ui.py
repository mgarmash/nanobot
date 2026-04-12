from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema
from nanobot.bus.events import OutboundMessage


def _iso_day_offset(day: str, delta_days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=delta_days)).isoformat()


def _agenda_buttons(day: str) -> list[list[dict[str, str]]]:
    return [
        [
            {"text": "⬅️", "data": f"calendar:agenda:prev:{day}"},
            {"text": "Today", "data": "calendar:agenda:today"},
            {"text": "➡️", "data": f"calendar:agenda:next:{day}"},
        ],
    ]


def _slots_buttons(day: str, duration_min: int) -> list[list[dict[str, str]]]:
    shorter = max(15, duration_min - 30)
    longer = min(180, duration_min + 30)
    return [
        [
            {"text": "Agenda", "data": f"calendar:agenda:show:{day}"},
            {"text": "⬅️", "data": f"calendar:agenda:prev:{day}"},
            {"text": "➡️", "data": f"calendar:agenda:next:{day}"},
        ],
        [
            {"text": f"{shorter}m", "data": f"calendar:slots:{day}:{shorter}"},
            {"text": f"{duration_min}m", "data": f"calendar:slots:{day}:{duration_min}"},
            {"text": f"{longer}m", "data": f"calendar:slots:{day}:{longer}"},
        ],
    ]


def _confirm_create_buttons(draft_token: str, calendar_kind: str) -> list[list[dict[str, str]]]:
    return [
        [
            {
                "text": "✅ Confirm",
                "data": f"calendar:confirm:create:{draft_token}:{calendar_kind}",
            },
        ],
        [
            {"text": "❌ Cancel", "data": "calendar:cancel"},
        ],
    ]


def _confirm_delete_buttons(event_id: str, calendar_kind: str) -> list[list[dict[str, str]]]:
    return [
        [
            {"text": "✅ Confirm", "data": f"calendar:confirm:delete:{event_id}:{calendar_kind}"},
        ],
        [
            {"text": "❌ Cancel", "data": "calendar:cancel"},
        ],
    ]


@tool_parameters(
    tool_parameters_schema(
        action=StringSchema(
            "Calendar UI action",
            enum=["agenda_day", "free_slots_day", "confirm_create", "confirm_delete"],
        ),
        content=StringSchema("Rendered text to send to the user"),
        day=StringSchema("ISO day for agenda/slot paging", nullable=True),
        duration_min=IntegerSchema(
            description="Slot duration in minutes", minimum=15, maximum=180, nullable=True
        ),
        draft_token=StringSchema("Draft token for create confirmation", nullable=True),
        event_id=StringSchema("Event identifier for delete confirmation", nullable=True),
        calendar_kind=StringSchema("Calendar kind alias or canonical value", nullable=True),
        required=["action", "content"],
    )
)
class CalendarUiTool(Tool):
    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False

    @property
    def name(self) -> str:
        return "calendar_ui"

    @property
    def description(self) -> str:
        return "Send calendar-specific Telegram UI messages with deterministic inline buttons for agenda paging and confirmations."

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        self._send_callback = callback

    def start_turn(self) -> None:
        self._sent_in_turn = False

    async def execute(self, **kwargs: Any) -> str:
        if not self._send_callback:
            return "Error: Calendar UI sending not configured"

        action = str(kwargs.get("action") or "").strip()
        content = str(kwargs.get("content") or "").strip()
        channel = str(kwargs.get("channel") or self._default_channel or "")
        chat_id = str(kwargs.get("chat_id") or self._default_chat_id or "")
        message_id = kwargs.get("message_id") or self._default_message_id
        day = kwargs.get("day")
        duration_min = int(kwargs.get("duration_min") or 30)
        draft_token = kwargs.get("draft_token")
        event_id = kwargs.get("event_id")
        calendar_kind = str(kwargs.get("calendar_kind") or "")
        edit_message_id = kwargs.get("edit_message_id")

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        if not content:
            return "Error: content is required"

        buttons: list[list[dict[str, str]]]
        if action == "agenda_day":
            if not day:
                return "Error: day is required for agenda_day"
            buttons = _agenda_buttons(str(day))
        elif action == "free_slots_day":
            if not day:
                return "Error: day is required for free_slots_day"
            buttons = _slots_buttons(str(day), duration_min)
        elif action == "confirm_create":
            if not draft_token or not calendar_kind:
                return "Error: draft_token and calendar_kind are required for confirm_create"
            buttons = _confirm_create_buttons(str(draft_token), calendar_kind)
        elif action == "confirm_delete":
            if not event_id or not calendar_kind:
                return "Error: event_id and calendar_kind are required for confirm_delete"
            buttons = _confirm_delete_buttons(str(event_id), calendar_kind)
        else:
            return f"Error: unknown action '{action}'"

        metadata: dict[str, Any]
        if message_id:
            metadata = {
                "message_id": message_id,
                "buttons": buttons,
            }
        else:
            metadata = {"buttons": buttons}
        if edit_message_id:
            metadata["_edit_message_id"] = edit_message_id

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            metadata=metadata,
        )
        await self._send_callback(msg)
        if channel == self._default_channel and chat_id == self._default_chat_id:
            self._sent_in_turn = True
        return f"Calendar UI sent to {channel}:{chat_id}"
