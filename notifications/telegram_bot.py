"""Simple Telegram notifier using BotFather tokens."""

from __future__ import annotations

import logging
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send messages to Telegram chats using a BotFather token."""

    def __init__(self, bot_token: str, chat_ids: Iterable[str | int]):
        if not bot_token:
            raise ValueError("Telegram bot token must be provided")
        self.bot_token = bot_token
        self.chat_ids = [str(chat_id) for chat_id in chat_ids if chat_id]
        self.client = httpx.AsyncClient(timeout=15.0)

    async def send_message(self, text: str, disable_preview: bool = True) -> None:
        """Broadcast a message to all configured chat IDs."""

        if not self.chat_ids:
            logger.warning("No chat IDs configured; skipping Telegram send")
            return

        payload = {
            "text": text,
            "disable_web_page_preview": disable_preview,
            "parse_mode": "Markdown",
        }

        for chat_id in self.chat_ids:
            try:
                await self._post(chat_id, payload)
            except Exception as exc:  # noqa: BLE001
                logger.error("Telegram send failed for chat %s: %s", chat_id, exc)

    async def _post(self, chat_id: str, payload: dict) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        await self.client.post(url, json={"chat_id": chat_id, **payload})

    async def aclose(self) -> None:
        await self.client.aclose()


def build_notifier_from_settings(settings) -> TelegramNotifier | None:
    """Create a TelegramNotifier from the global settings object."""

    chat_ids = [chat_id.strip() for chat_id in settings.telegram_chat_ids.split(",") if chat_id.strip()]
    if not settings.telegram_bot_token or not chat_ids:
        return None
    return TelegramNotifier(settings.telegram_bot_token, chat_ids)
