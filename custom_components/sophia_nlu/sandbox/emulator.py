"""Sandbox home emulation for the Sophia NLU conversation agent.

Mirrors aquila/home_emulator.py but runs inside a live Home Assistant
instance.  When the user sets ``sophia_nlu: yaml_file: /path/to/home_config.yaml``
in configuration.yaml the agent wipes the existing floors/areas/entities
and rebuilds the world from that YAML.

Usage (from conversation.py):
    from .sandbox import async_emulate_sandbox_home
    await async_emulate_sandbox_home(hass, yaml_file)

This module is deliberately self-contained — it does not import from the
test-suite's aquila package so the integration can be copied standalone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import dataclasses
import logging

import yaml

from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TOGGLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
)
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.components.intent.timers import (
    async_register_timer_handler,
    TimerEventType,
    TimerInfo,
)
from homeassistant.components.todo import (
    TodoItem,
    TodoListEntity,
    TodoListEntityFeature,
)
from homeassistant.components.todo.const import DATA_COMPONENT as TODO_DATA
from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
)
from homeassistant.components.media_player.const import (
    DOMAIN as MEDIA_PLAYER_DOMAIN,
)
from homeassistant.helpers.entity_component import EntityComponent

from ..const import DOMAIN as SOPHIA_DOMAIN

_LOGGER = logging.getLogger(__name__)

# Must match the value used in aquila and in tests.
TIMER_DEVICE_ID = "aquila_test_device"

_SHOPPING_LIST_KEY = "aquila_shopping_list"
_TODO_LISTS_KEY = "aquila_todo_lists"
_SCENE_REG_KEY = "aquila_scene_registry"
_SCRIPT_REG_KEY = "aquila_script_registry"

# Services to mock per domain: (service_name, resulting_state, attr_passthrough)
_DOMAIN_SERVICES: dict[str, list[tuple[str, str | None, bool]]] = {
    "light": [
        (SERVICE_TURN_ON, "on", True),
        (SERVICE_TURN_OFF, "off", False),
    ],
    "switch": [
        (SERVICE_TURN_ON, "on", False),
        (SERVICE_TURN_OFF, "off", False),
    ],
    "fan": [
        (SERVICE_TURN_ON, "on", True),
        (SERVICE_TURN_OFF, "off", False),
        ("set_percentage", None, True),
    ],
    "media_player": [
        (SERVICE_TURN_ON, "on", False),
        (SERVICE_TURN_OFF, "off", False),
        ("media_pause", "paused", False),
        ("media_play", "playing", False),
    ],
    "cover": [
        ("open_cover", "open", False),
        ("close_cover", "closed", False),
        ("set_cover_position", None, True),
    ],
    "lock": [
        ("lock", "locked", False),
        ("unlock", "unlocked", False),
    ],
    "climate": [
        ("set_temperature", None, True),
        ("set_hvac_mode", None, True),
    ],
}


# ---------------------------------------------------------------------------
# Service handlers (same semantics as aquila/home_emulator.py)
# ---------------------------------------------------------------------------


def _make_service_handler(
    hass: HomeAssistant,
    new_state: str | None,
    pass_attrs: bool,
):
    async def handler(call: ServiceCall) -> None:
        entity_ids = call.data.get(ATTR_ENTITY_ID) or []
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        extra_attrs = {k: v for k, v in call.data.items() if k != ATTR_ENTITY_ID}
        # Convert brightness_pct (0-100) to brightness (0-255) like the real light component.
        if "brightness_pct" in extra_attrs:
            extra_attrs["brightness"] = round(
                255 * float(extra_attrs.pop("brightness_pct")) / 100
            )
        for eid in entity_ids:
            current = hass.states.get(eid)
            if current is None:
                _LOGGER.debug("Service call target %s has no state, skipping", eid)
                continue
            attrs = dict(current.attributes)
            if pass_attrs:
                attrs.update(extra_attrs)
            state = new_state if new_state is not None else current.state
            hass.states.async_set(eid, state, attrs)

    return handler


def _make_volume_mute_handler(hass: HomeAssistant):
    async def handler(call: ServiceCall) -> None:
        entity_ids = call.data.get(ATTR_ENTITY_ID, [])
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        raw_muted = call.data.get("is_volume_muted", False)
        muted_str = "true" if raw_muted else "false"
        for eid in entity_ids:
            current = hass.states.get(eid)
            attrs = dict(current.attributes) if current else {}
            attrs["is_volume_muted"] = muted_str
            state = current.state if current else "on"
            hass.states.async_set(eid, state, attrs)

    return handler


def _make_volume_set_handler(hass: HomeAssistant):
    async def handler(call: ServiceCall) -> None:
        entity_ids = call.data.get(ATTR_ENTITY_ID, [])
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        raw_level = call.data.get("volume_level", 0.0)
        fval = float(raw_level)
        if isinstance(raw_level, int) or fval > 1.0:
            level_int = round(fval)
            normalized = fval / 100.0
        else:
            level_int = round(fval * 100)
            normalized = fval
        for eid in entity_ids:
            current = hass.states.get(eid)
            attrs = dict(current.attributes) if current else {}
            attrs["volume_level"] = level_int
            state = current.state if current else "on"
            hass.states.async_set(eid, state, attrs)
            component: EntityComponent | None = hass.data.get(MEDIA_PLAYER_DOMAIN)
            if component:
                for entity in component.entities:
                    if entity.entity_id == eid:
                        entity._attr_volume_level = normalized  # noqa: SLF001
                        break

    return handler


def _make_toggle_handler(hass: HomeAssistant):
    async def handler(call: ServiceCall) -> None:
        entity_ids = call.data.get(ATTR_ENTITY_ID, [])
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        for eid in entity_ids:
            current = hass.states.get(eid)
            attrs = dict(current.attributes) if current else {}
            cur_state = current.state if current else "off"
            domain = eid.split(".", 1)[0] if "." in eid else ""
            if domain == "cover":
                new_state = "closed" if cur_state == "open" else "open"
            elif domain == "lock":
                new_state = "unlocked" if cur_state == "locked" else "locked"
            elif domain == "valve":
                new_state = "closed" if cur_state == "open" else "open"
            else:
                new_state = "off" if cur_state.lower() == "on" else "on"
            hass.states.async_set(eid, new_state, attrs)

    return handler


def _make_homeassistant_toggle_handler(hass: HomeAssistant):
    toggle_handler = _make_toggle_handler(hass)

    async def handler(call: ServiceCall) -> None:
        await toggle_handler(call)

    return handler

def _register_domain_services(hass: HomeAssistant, domain: str) -> None:
    for service_name, new_state, pass_attrs in _DOMAIN_SERVICES.get(domain, []):
        # Force-register (overwrite) so the real component handlers are replaced
        hass.services.async_register(
            domain,
            service_name,
            _make_service_handler(hass, new_state, pass_attrs),
        )

    if domain == "media_player":
        hass.services.async_register(
            "media_player", "volume_mute", _make_volume_mute_handler(hass)
        )
        hass.services.async_register(
            "media_player", "volume_set", _make_volume_set_handler(hass)
        )

    if domain in (
        "light", "switch", "fan", "cover", "lock", "valve",
        "media_player", "climate", "input_boolean", "scene", "script",
    ):
        hass.services.async_register(domain, SERVICE_TOGGLE, _make_toggle_handler(hass))

    # homeassistant.turn_on / turn_off / toggle are also commonly used by intents
    if not hass.services.has_service("homeassistant", SERVICE_TOGGLE):
        hass.services.async_register(
            "homeassistant", SERVICE_TOGGLE, _make_homeassistant_toggle_handler(hass)
        )


def _make_scene_handler(hass: HomeAssistant, entities: dict[str, dict[str, Any]]):
    async def handler(call: ServiceCall) -> None:
        for entity_id, target in entities.items():
            current = hass.states.get(entity_id)
            attrs = dict(current.attributes) if current else {}
            new_state = str(target.get("state", current.state if current else "on"))
            extra = {k: v for k, v in target.items() if k != "state"}
            attrs.update(extra)
            hass.states.async_set(entity_id, new_state, attrs)

    return handler


def _make_script_handler(hass: HomeAssistant, actions: list[dict[str, Any]]):
    async def handler(call: ServiceCall) -> None:
        for action in actions:
            svc = action.get("action") or action.get("service", "")
            if not svc or "." not in svc:
                continue
            svc_domain, svc_name = svc.split(".", 1)
            data: dict[str, Any] = dict(action.get("data", {}))
            target: dict[str, Any] = action.get("target", {})
            entity_ids = target.get("entity_id") or data.pop("entity_id", [])
            if isinstance(entity_ids, str):
                entity_ids = [entity_ids]
            if entity_ids:
                data[ATTR_ENTITY_ID] = entity_ids
            await hass.services.async_call(svc_domain, svc_name, data, blocking=True)

    return handler


# ---------------------------------------------------------------------------
# Todo / MediaPlayer entities (mirrors aquila)
# ---------------------------------------------------------------------------


class AquilaTodoListEntity(TodoListEntity):
    """Minimal in-memory TodoListEntity for sandbox emulation."""

    _attr_supported_features = (
        TodoListEntityFeature.CREATE_TODO_ITEM
        | TodoListEntityFeature.UPDATE_TODO_ITEM
        | TodoListEntityFeature.DELETE_TODO_ITEM
    )

    def __init__(self, list_name: str, list_id: str) -> None:
        self._attr_name = list_name
        self._attr_unique_id = f"aquila_todo_{list_id}"
        self._items: list[TodoItem] = []

    @property
    def todo_items(self) -> list[TodoItem]:
        return list(self._items)

    async def async_create_todo_item(self, item: TodoItem) -> None:
        import uuid

        self._items.append(dataclasses.replace(item, uid=item.uid or str(uuid.uuid4())))
        self.async_write_ha_state()

    async def async_update_todo_item(self, item: TodoItem) -> None:
        for i, existing in enumerate(self._items):
            if existing.uid == item.uid:
                self._items[i] = dataclasses.replace(
                    existing,
                    status=item.status if item.status is not None else existing.status,
                    summary=item.summary if item.summary is not None else existing.summary,
                )
                break
        self.async_write_ha_state()

    async def async_delete_todo_items(self, uids: list[str]) -> None:
        uid_set = set(uids)
        self._items = [it for it in self._items if it.uid not in uid_set]
        self.async_write_ha_state()


class AquilaMediaPlayerEntity(MediaPlayerEntity):
    """Minimal MediaPlayerEntity so volume intents can call entity methods."""

    _attr_supported_features = (
        MediaPlayerEntityFeature.VOLUME_SET
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.PAUSE
        | MediaPlayerEntityFeature.PLAY
    )

    def __init__(self, entity_id: str, name: str, initial_volume: float = 0.5) -> None:
        self._attr_unique_id = f"aquila_mp_{entity_id}"
        self._attr_name = name
        self.entity_id = entity_id
        self._attr_volume_level: float = initial_volume

    @property
    def volume_level(self) -> float | None:
        return self._attr_volume_level

    async def async_set_volume_level(self, volume: float) -> None:
        self._attr_volume_level = volume
        current = self.hass.states.get(self.entity_id)
        attrs = dict(current.attributes) if current else {}
        attrs["volume_level"] = round(volume * 100)
        state = current.state if current else "on"
        self.hass.states.async_set(self.entity_id, state, attrs)

    async def async_mute_volume(self, mute: bool) -> None:
        current = self.hass.states.get(self.entity_id)
        attrs = dict(current.attributes) if current else {}
        attrs["is_volume_muted"] = "true" if mute else "false"
        state = current.state if current else "on"
        self.hass.states.async_set(self.entity_id, state, attrs)


def _shopping_list(hass: HomeAssistant) -> list[dict]:
    if _SHOPPING_LIST_KEY not in hass.data:
        hass.data[_SHOPPING_LIST_KEY] = []
    return hass.data[_SHOPPING_LIST_KEY]


def _scene_registry(hass: HomeAssistant) -> dict[str, dict]:
    if _SCENE_REG_KEY not in hass.data:
        hass.data[_SCENE_REG_KEY] = {}
    return hass.data[_SCENE_REG_KEY]


def _script_registry(hass: HomeAssistant) -> dict[str, list]:
    if _SCRIPT_REG_KEY not in hass.data:
        hass.data[_SCRIPT_REG_KEY] = {}
    return hass.data[_SCRIPT_REG_KEY]


def _make_scene_router(hass: HomeAssistant):
    async def handler(call: ServiceCall) -> None:
        entity_ids = call.data.get(ATTR_ENTITY_ID, [])
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        registry = _scene_registry(hass)
        for eid in entity_ids:
            entities = registry.get(eid, {})
            await _make_scene_handler(hass, entities)(call)

    return handler


def _make_script_router(hass: HomeAssistant):
    async def handler(call: ServiceCall) -> None:
        entity_ids = call.data.get(ATTR_ENTITY_ID, [])
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        registry = _script_registry(hass)
        for eid in entity_ids:
            actions = registry.get(eid, [])
            await _make_script_handler(hass, actions)(call)

    return handler


# ---------------------------------------------------------------------------
# Wipe + load helpers
# ---------------------------------------------------------------------------


def _is_protected_entity(hass: HomeAssistant, entity_id: str) -> bool:
    """Return True for entities that must survive a sandbox wipe."""
    # Keep the sophia conversation agent itself and core HA infra.
    if entity_id == f"conversation.{SOPHIA_DOMAIN}":
        return True
    if entity_id.startswith("conversation."):
        # Preserve any conversation entity (e.g. default agent)
        return True
    # Registry-level check: platform == sophia_nlu
    try:
        entity_reg = er.async_get(hass)
        entry = entity_reg.entities.get(entity_id)
        if entry is not None and entry.platform == SOPHIA_DOMAIN:
            return True
    except Exception:
        pass
    return False


def _is_protected_device(hass: HomeAssistant, device_id: str) -> bool:
    """Return True for devices that must survive a sandbox wipe."""
    try:
        device_reg = dr.async_get(hass)
        device = device_reg.devices.get(device_id)
        if device is None:
            return False
        for domain, _ident in device.identifiers:
            if domain == SOPHIA_DOMAIN:
                return True
    except Exception:
        pass
    return False


@callback
def _wipe_existing_home(hass: HomeAssistant) -> None:
    """Remove existing floors, areas, entities and states before emulation.

    Mirrors what ``demo:`` does but focused on the emulated home.  Protected
    entities/devices belonging to ``sophia_nlu`` are preserved so the
    conversation agent does not delete itself.
    """
    entity_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass)
    device_reg = dr.async_get(hass)

    # 1. Remove states (keep protected conversation states)
    for state in list(hass.states.async_all()):
        if _is_protected_entity(hass, state.entity_id):
            continue
        hass.states.async_remove(state.entity_id)

    # 2. Remove entity registry entries (keep protected)
    for entity_id in list(entity_reg.entities.keys()):
        if _is_protected_entity(hass, entity_id):
            continue
        entity_reg.async_remove(entity_id)

    # 3. Remove devices (keep sophia device)
    for device_id in list(device_reg.devices.keys()):
        if _is_protected_device(hass, device_id):
            continue
        try:
            device_reg.async_remove_device(device_id)
        except KeyError:
            # Already removed via cascade or race
            pass

    # 4. Delete areas
    for area_id in list(area_reg.areas.keys()):
        try:
            area_reg.async_delete(area_id)
        except KeyError:
            pass

    # 5. Delete floors
    for floor_id in list(floor_reg.floors.keys()):
        try:
            floor_reg.async_delete(floor_id)
        except KeyError:
            pass

    # 6. Clear sandbox-private hass.data keys so stale registries do not leak
    hass.data.pop(_SHOPPING_LIST_KEY, None)
    hass.data.pop(_TODO_LISTS_KEY, None)
    hass.data.pop(_SCENE_REG_KEY, None)
    hass.data.pop(_SCRIPT_REG_KEY, None)
    # Also clear any leaked timer handler registration — it will be re-added
    # during emulation if the new config contains timers.


def _resolve_yaml_path(hass: HomeAssistant, yaml_file: str) -> Path:
    """Resolve yaml_file to an absolute Path.

    Absolute paths are used as-is.  Relative paths are resolved relative to
    ``hass.config.config_dir`` (the directory containing configuration.yaml),
    matching HA's own include semantics.  ``~`` is expanded.
    """
    p = Path(yaml_file).expanduser()
    if not p.is_absolute():
        try:
            config_dir = Path(hass.config.config_dir)
        except Exception:
            config_dir = Path.cwd()
        p = (config_dir / p).resolve()
    return p


def _load_home_config(yaml_path: Path) -> dict[str, Any]:
    """Load and parse a home_config.yaml file.

    Synchronous — must be called via hass.async_add_executor_job
    to avoid blocking the event loop. See:
    https://developers.home-assistant.io/docs/asyncio_blocking_operations/#open
    """
    if not yaml_path.is_file():
        raise FileNotFoundError(f"home_config.yaml not found: {yaml_path}")
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"home_config.yaml did not parse to a dict: {yaml_path}")
    return data


async def _setup_shopping_list(hass: HomeAssistant) -> None:
    """Create an in-memory ShoppingData backed by hass.data."""
    try:
        from homeassistant.components.shopping_list.common import (
            DOMAIN as SL_DOMAIN,
            ShoppingData,
        )
        from homeassistant.components.shopping_list import intent as sl_intent
        from homeassistant.config_entries import ConfigEntry
    except ImportError:
        _LOGGER.debug("shopping_list component not available, skipping")
        return

    class _AquilaShoppingData(ShoppingData):
        def __init__(self, hass: HomeAssistant) -> None:
            self.hass = hass

        @property
        def items(self) -> list[dict]:
            return _shopping_list(self.hass)

        @items.setter
        def items(self, value: list[dict]) -> None:
            self.hass.data[_SHOPPING_LIST_KEY] = value

        async def async_add(self, name: str) -> dict:
            import uuid

            item = {"name": name, "id": uuid.uuid4().hex, "complete": False}
            self.items.append(item)
            self.hass.bus.async_fire("shopping_list_updated")
            return item

        async def async_complete(self, name: str) -> list[dict]:
            completed = []
            for item in self.items:
                if item["name"].lower() == name.lower() and not item["complete"]:
                    item["complete"] = True
                    completed.append(item)
            self.hass.bus.async_fire("shopping_list_updated")
            return completed

        def save(self) -> None:
            pass

    # Remove any previous aquila shopping_list entry to avoid duplicates
    for eid, entry in list(hass.config_entries._entries.items()):  # type: ignore[attr-defined]
        if getattr(entry, "domain", None) == SL_DOMAIN and getattr(
            entry, "unique_id", None
        ) == "aquila_shopping_list":
            hass.config_entries._entries.pop(eid, None)  # type: ignore[attr-defined]

    entry = ConfigEntry(
        version=1,
        minor_version=1,
        domain=SL_DOMAIN,
        title="Shopping List",
        data={},
        options={},
        source="aquila",
        unique_id="aquila_shopping_list",
        discovery_keys={},
        subentries_data=None,
    )
    entry._async_set_state(hass, "loaded", None)
    entry.runtime_data = _AquilaShoppingData(hass)
    hass.config_entries._entries[entry.entry_id] = entry  # type: ignore[attr-defined]
    # Only register shopping_list intents if not already registered by the
    # core shopping_list integration — otherwise HA logs
    # "Intent HassShoppingList* is being overwritten".
    from homeassistant.helpers.intent import DATA_KEY as INTENT_DATA_KEY

    intents = hass.data.get(INTENT_DATA_KEY, {})
    if not any(
        k in intents
        for k in (
            sl_intent.INTENT_ADD_ITEM,
            sl_intent.INTENT_COMPLETE_ITEM,
            sl_intent.INTENT_LAST_ITEMS,
        )
    ):
        await sl_intent.async_setup_intents(hass)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def async_emulate_sandbox_home(
    hass: HomeAssistant, yaml_file: str
) -> dict[str, str]:
    """Wipe the current HA home and rebuild it from *yaml_file*.

    Args:
        hass: The HomeAssistant instance.
        yaml_file: Absolute or config-dir-relative path to a
            ``home_config.yaml`` file (one of the ``datasets/<suite>/``
            files).

    Returns:
        area_map: ``{yaml_area_id: ha_area_registry_id}`` mapping, identical
        to ``aquila.home_emulator.emulate_home`` so adapters and tests can
        reuse it if needed.
    """
    yaml_path = _resolve_yaml_path(hass, yaml_file)
    _LOGGER.info("Sandbox emulation: loading home config from %s", yaml_path)

    # Wipe stale registry entries first so HA does not try to call services on
    # entities from a previous run while the new config is loading.
    _wipe_existing_home(hass)

    config = await hass.async_add_executor_job(_load_home_config, yaml_path)

    for svc, state, pass_attrs in [
        (SERVICE_TURN_ON, "on", True),
        (SERVICE_TURN_OFF, "off", False),
    ]:
        hass.services.async_register(
            "homeassistant",
            svc,
            _make_service_handler(hass, state, pass_attrs),
        )

    floor_reg = fr.async_get(hass)
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)

    floor_map: dict[str, str] = {}
    for floor in config.get("floors", []):
        entry = floor_reg.async_create(
            floor["name"],
            level=floor.get("level"),
        )
        floor_map[floor["id"]] = entry.floor_id

    area_map: dict[str, str] = {}
    for area in config.get("areas", []):
        floor_ha_id = floor_map.get(area.get("floor"))
        entry = area_reg.async_create(area["name"], floor_id=floor_ha_id)
        area_map[area["id"]] = entry.id

    for device in config.get("devices", []):
        entity_id: str = device["id"]
        domain, object_id = entity_id.split(".", 1)

        reg_entry = entity_reg.async_get_or_create(
            domain,
            "aquila_emulator",
            unique_id=entity_id,
            suggested_object_id=object_id,
            original_name=device["name"],
        )

        ha_area_id = area_map.get(device.get("area_id"))
        if ha_area_id:
            entity_reg.async_update_entity(reg_entry.entity_id, area_id=ha_area_id)

        attrs = dict(device.get("attributes") or {})
        attrs.setdefault("friendly_name", device["name"])
        hass.states.async_set(
            reg_entry.entity_id, str(device.get("state", "off")), attrs
        )

        async_expose_entity(hass, "assist", reg_entry.entity_id, True)
        _register_domain_services(hass, domain)

        if domain == "media_player":
            initial_volume_raw = attrs.get("volume_level", 50)
            initial_volume = (
                float(initial_volume_raw) / 100.0
                if float(initial_volume_raw) > 1.0
                else float(initial_volume_raw)
            )
            mp_entity = AquilaMediaPlayerEntity(
                reg_entry.entity_id, device["name"], initial_volume
            )
            mp_entity.hass = hass
            component: EntityComponent = hass.data.setdefault(
                MEDIA_PLAYER_DOMAIN,
                EntityComponent(
                    __import__("logging").getLogger(__name__),
                    MEDIA_PLAYER_DOMAIN,
                    hass,
                ),
            )
            await component.async_add_entities([mp_entity])

    for scene in config.get("scenes", []):
        scene_entity_id = f"scene.{scene['id']}"
        entities: dict[str, dict] = scene.get("entities", {})
        reg_entry = entity_reg.async_get_or_create(
            "scene",
            "aquila_emulator",
            unique_id=scene_entity_id,
            suggested_object_id=scene["id"],
            original_name=scene["name"],
        )
        attrs = {"friendly_name": scene["name"]}
        hass.states.async_set(reg_entry.entity_id, "scening", attrs)
        async_expose_entity(hass, "assist", reg_entry.entity_id, True)
        if not hass.services.has_service("scene", SERVICE_TURN_ON):
            hass.services.async_register(
                "scene",
                SERVICE_TURN_ON,
                _make_scene_router(hass),
            )
        _scene_registry(hass)[reg_entry.entity_id] = entities

    for script in config.get("scripts", []):
        script_entity_id = f"script.{script['id']}"
        actions: list[dict] = script.get("actions", [])
        reg_entry = entity_reg.async_get_or_create(
            "script",
            "aquila_emulator",
            unique_id=script_entity_id,
            suggested_object_id=script["id"],
            original_name=script["name"],
        )
        attrs = {"friendly_name": script["name"]}
        hass.states.async_set(reg_entry.entity_id, "off", attrs)
        async_expose_entity(hass, "assist", reg_entry.entity_id, True)
        if not hass.services.has_service("script", SERVICE_TURN_ON):
            hass.services.async_register(
                "script",
                SERVICE_TURN_ON,
                _make_script_router(hass),
            )
        _script_registry(hass)[reg_entry.entity_id] = actions

    for timer in config.get("timers", []):
        entity_id = timer["id"] if "." in timer["id"] else f"timer.{timer['id']}"
        attrs = {"friendly_name": timer["name"]}
        hass.states.async_set(entity_id, timer.get("state", "idle"), attrs)

    if config.get("timers"):

        @callback
        def _noop_timer_handler(event_type: TimerEventType, timer: TimerInfo) -> None:
            pass

        async_register_timer_handler(hass, TIMER_DEVICE_ID, _noop_timer_handler)

    hass.data[_SHOPPING_LIST_KEY] = []
    await _setup_shopping_list(hass)

    hass.data.setdefault(_TODO_LISTS_KEY, {})
    for lst in config.get("lists", []):
        list_name = lst["name"]
        list_id = lst.get("id", list_name.lower().replace(" ", "_"))
        entity = AquilaTodoListEntity(list_name, list_id)
        reg_entry = entity_reg.async_get_or_create(
            "todo",
            "aquila_emulator",
            unique_id=f"aquila_todo_{list_id}",
            suggested_object_id=list_id,
            original_name=list_name,
        )
        entity.hass = hass
        entity.entity_id = reg_entry.entity_id
        hass.states.async_set(reg_entry.entity_id, "0", {"friendly_name": list_name})
        async_expose_entity(hass, "assist", reg_entry.entity_id, True)
        hass.data[_TODO_LISTS_KEY][list_name.lower()] = entity
        todo_component = hass.data.get(TODO_DATA)
        if todo_component is not None:
            await todo_component.async_add_entities([entity])

    _LOGGER.info(
        "Sandbox emulation complete: %d floors, %d areas, %d devices",
        len(floor_map),
        len(area_map),
        len(config.get("devices", [])),
    )
    return area_map
