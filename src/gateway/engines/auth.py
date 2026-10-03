"""
Auth & Token Lifecycle Manager
Handles OAuth token lifecycle, refresh races, and credential security.
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TokenState:
    access_token: str
    refresh_token: str | None
    expires_at: datetime
    provider: str

class AuthLifecycleManager:
    """
    Solves Category 4: Authentication & token lifecycle.
    Prevents refresh races and ensures tokens are valid before outbound calls.
    """
    def __init__(self) -> None:
        self._tokens: dict[str, TokenState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def get_valid_token(
        self, 
        provider: str, 
        identity_key: str, 
        refresh_fn: Callable[[str | None], Awaitable[TokenState]]
    ) -> str:
        """
        Get a valid token, refreshing if necessary. 
        Uses asyncio.Lock to prevent concurrent requests from triggering duplicate refresh calls.
        """
        key = f"{provider}:{identity_key}"
        lock = self._get_lock(key)
            
        async with lock:
            state = self._tokens.get(key)
            
            # 60 second safety buffer before expiry
            now = datetime.now(UTC)
            if not state or state.expires_at < now + timedelta(seconds=60):
                logger.info("Refreshing token for %s", key)
                old_refresh = state.refresh_token if state else None
                new_state = await refresh_fn(old_refresh)
                self._tokens[key] = new_state
                state = new_state
                
            return state.access_token

    def force_invalidate(self, provider: str, identity_key: str) -> None:
        """Manually invalidate a token (e.g. if the provider unexpectedly rejects it)."""
        key = f"{provider}:{identity_key}"
        if key in self._tokens:
            del self._tokens[key]
