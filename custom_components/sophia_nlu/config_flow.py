"""Config flow for the Sophia NLU integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.helpers.network import async_get_source_ip

from .const import CONF_HOST, CONF_PORT, DEFAULT_HOST, DEFAULT_PORT, DOMAIN

_LOGGER = logging.getLogger(__name__)


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

        # Use the HA host's own IP as the default instead of 127.0.0.1,
        # since the NLU app typically runs on the same machine.
        default_host = DEFAULT_HOST
        try:
            source_ip = await async_get_source_ip(self.hass)
            if source_ip:
                default_host = source_ip
        except Exception:
            _LOGGER.debug(
                "Could not determine local IP, falling back to %s", DEFAULT_HOST
            )

        data_schema = vol.Schema(
            {
                vol.Required(CONF_HOST, default=default_host): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
            }
        )

        return self.async_show_form(
            step_id="user", data_schema=data_schema, errors=errors
        )
