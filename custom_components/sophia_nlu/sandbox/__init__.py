"""Sandbox package for the Sophia NLU integration.

Exposes :func:`async_emulate_sandbox_home` — the runtime equivalent of
``aquila.home_emulator.emulate_home`` — for use by the conversation agent
when ``sophia_nlu: yaml_file: ...`` is present in configuration.yaml.
"""

from .emulator import (
    AquilaMediaPlayerEntity,
    AquilaTodoListEntity,
    TIMER_DEVICE_ID,
    async_emulate_sandbox_home,
)

__all__ = [
    "AquilaMediaPlayerEntity",
    "AquilaTodoListEntity",
    "TIMER_DEVICE_ID",
    "async_emulate_sandbox_home",
]
