
from __future__ import annotations
import asyncio
import logging
import socket
from typing import Any
import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.core import HomeAssistant
from .const import CONF_HOST, CONF_PORT, DEFAULT_PORT, DOMAIN

_LOGGER = logging.getLogger(__name__)

CONF_HOST = "host"
CONF_PORT = "port"


def _get_local_ip() -> str:
    """Determine the local IP address using a UDP trick (no packet sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))   # doesn't actually send anything
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


async def _async_validate_connection(host: str, port: int) -> bool:
    """Validate that we can connect to the NLU server."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=5.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


class SophiaNLUConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Sophia NLU."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        # Resolve the default host dynamically at form-render time
        default_host = await self.hass.async_add_executor_job(_get_local_ip)

        step_user_data_schema = vol.Schema(
            {
                vol.Required(CONF_HOST, default=default_host): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
            }
        )

        if user_input is not None:
            host = user_input[CONF_HOST]
            port = user_input[CONF_PORT]
            if not await _async_validate_connection(host, port):
                errors["base"] = "cannot_connect"
            else:
                await self.async_set_unique_id(f"{host}:{port}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"Sophia NLU ({host}:{port})", data=user_input
                )

        return self.async_show_form(
            step_id="user", data_schema=step_user_data_schema, errors=errors
        )




