"""The conversation platform for the Sophia NLU integration."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Callable
from datetime import date
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    device_registry as dr,
    floor_registry as fr,
    intent,
)


from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util, ulid

from .const import CONF_HOST, CONF_PORT, DOMAIN, get_service_call, is_standard_intent

_LOGGER = logging.getLogger(__name__)

# Sentinel used by _sanitize() to signal a value cannot be JSON-serialized
_UNSERIALIZABLE = object()

# Alarm intents handled directly via alarm_control_panel service calls
ALARM_DOMAIN = "alarm_control_panel"
ALARM_INTENTS = frozenset(["HassAlarmSetMode", "HassAlarmDisarm", "HassAlarmTrigger"])
ALARM_FIXED_SERVICES: dict[str, str] = {"HassAlarmDisarm": "alarm_disarm", "HassAlarmTrigger": "alarm_trigger"}
ALARM_MODE_SERVICES: dict[str, str] = {"home": "alarm_arm_home", "away": "alarm_arm_away", "night": "alarm_arm_night", "vacation": "alarm_arm_vacation", "custom_bypass": "alarm_arm_custom_bypass", "custom": "alarm_arm_custom_bypass"}

# Vacuum clean_area and media search intents need more than a single service call with the slots passed straight through
VACUUM_DOMAIN = "vacuum"
VACUUM_CLEAN_AREA_FEATURE = 16384
MEDIA_PLAYER_DOMAIN = "media_player"
MEDIA_SEARCH_FEATURES = 4194304 | 512  # SEARCH_MEDIA | PLAY_MEDIA

# Max seconds to wait for a device to report its new state after a service call (covers device failures / offline devices)
SETTLE_TIMEOUT = 1.75


def _light_brightness_pct(state_obj: State) -> int | None:
    """Return a light's brightness as a rounded percent, or None if it has no brightness attribute."""
    raw = state_obj.attributes.get("brightness")
    if raw is None:
        return None
    try:
        return round(float(raw) / 255.0 * 100.0)
    except (TypeError, ValueError):
        return None


def _build_settle_predicate(service_domain: str, service_data: dict[str, Any], before: State | None) -> Callable[[State], bool] | None:
    """Return a predicate that is True once the entity has reported the result of the service call, or None if there's nothing worth waiting for.

    Some integrations (Zigbee, Z-Wave, Matter, cloud) only update the HA state machine when the device confirms, which can be after the blocking service call has already returned.  Reading the state straight away would then return the previous value.
    Only brightness changes on lights are handled for now.
    """
    if service_domain != "light" or before is None:
        return None
    if before.state in ("unavailable", "unknown") or before.attributes.get("assumed_state"):
        return None

    # Absolute brightness: settled when the reported brightness is at (or within 1% of) the target, or the light is off for 0%
    if "brightness_pct" in service_data:
        try:
            target = int(service_data["brightness_pct"])
        except (TypeError, ValueError):
            return None

        def absolute_settled(state_obj: State) -> bool:
            if target <= 0:
                return state_obj.state == "off"
            pct = _light_brightness_pct(state_obj)
            return state_obj.state == "on" and pct is not None and abs(pct - target) <= 1

        # Already at the target, so no state change will ever arrive
        return None if absolute_settled(before) else absolute_settled

    # Relative brightness: the target is unknown (clamping / device quantization), so settled means the reported state changed
    if "brightness_step_pct" in service_data:
        try:
            step = int(service_data["brightness_step_pct"])
        except (TypeError, ValueError):
            return None

        before_pct = _light_brightness_pct(before)
        if step < 0 and before.state == "off":
            return None
        if step > 0 and before_pct is not None and before_pct >= 100:
            return None
        before_state = before.state

        def relative_settled(state_obj: State) -> bool:
            return state_obj.state != before_state or _light_brightness_pct(state_obj) != before_pct

        return relative_settled

    return None


class _SettleWatch:
    """Watches an entity for the state_changed event that shows a service call has taken effect.

    Must be created BEFORE the service call is made, otherwise an event fired during the (blocking) call is missed.  Waiting returns immediately if the state is already settled, so fast devices pay no extra latency.
    """

    def __init__(self, hass: HomeAssistant, entity_id: str, predicate: Callable[[State], bool], timeout: float = SETTLE_TIMEOUT) -> None:
        self._hass = hass
        self._entity_id = entity_id
        self._predicate = predicate
        self._timeout = timeout
        self._done = asyncio.Event()
        self._unsub: Callable[[], None] | None = async_track_state_change_event(hass, [entity_id], self._on_change)

    @callback
    def _on_change(self, event: Event) -> None:
        new_state = event.data.get("new_state")
        if new_state is not None and self._predicate(new_state):
            self._done.set()

    async def wait(self) -> bool:
        """Wait for the settled state.  Returns False if it timed out."""
        current = self._hass.states.get(self._entity_id)
        if current is not None and self._predicate(current):
            return True

        started = time.monotonic()
        try:
            await asyncio.wait_for(self._done.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            _LOGGER.debug("%s did not report a settled state within %.2fs, using current state", self._entity_id, self._timeout)
            return False
        _LOGGER.debug("%s settled after %.3fs", self._entity_id, time.monotonic() - started)
        return True

    def cancel(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities, emulating home if yaml_file is configured."""
    yaml_file = hass.data.get(DOMAIN, {}).get("yaml_file")

    entity_reg = er.async_get(hass)
    existing_id = entity_reg.async_get_entity_id("conversation", DOMAIN, config_entry.entry_id)
    if existing_id == f"conversation.{DOMAIN}":
        entity_reg.async_update_entity(existing_id, new_entity_id="conversation.sophia_nlu_engine")

    entity = SophiaNLUConversationEntity(config_entry)
    if yaml_file:
        from .sandbox import async_emulate_sandbox_home
        try:
            await async_emulate_sandbox_home(hass, yaml_file)
        except FileNotFoundError as err:
            _LOGGER.error("Sandbox yaml_file not found: %s", err)
        except Exception as err:
            _LOGGER.exception("Failed to emulate sandbox home from %s: %s", yaml_file, err)

    async_add_entities([entity])

class SophiaNLUConversationEntity(conversation.ConversationEntity):
    """Sophia NLU conversation agent entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = False
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._host = entry.data[CONF_HOST]
        self._port = entry.data[CONF_PORT]
        self.emulated_home = None

        self._attr_unique_id = entry.entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self.entry.entry_id)},
            name="Sophia NLU Engine",
            manufacturer="Aquila Labs",
        )
        self._conversation_sessions: dict[str, dict] = {}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return ["en"]

    @property
    def attribution(self) -> conversation.AgentAttribution | None:
        """Return attribution for conversation agent."""
        return "Sophia NLU (https://nlu.to/ha/)"

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _is_entity_usable(self, entity_entry: er.RegistryEntry) -> bool:
        """Return True if the entity is enabled and exposed to conversation."""
        # 1. Must not be disabled
        if entity_entry.disabled_by is not None:
            return False

        # 2. Must be exposed to the conversation agent
        conversation_opts = entity_entry.options.get("conversation", {})
        if not conversation_opts.get("should_expose", False):
            return False

        return True

    async def _async_send_payload(self, payload_dict: dict[str, Any]) -> list[dict]:
        """Send a raw JSON payload over TCP to the NLU server and return parsed JSON responses."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port), timeout=5.0
        )
        try:
            request = json.dumps(payload_dict, separators=(",", ":"))
            payload = f"{request}\n"
            writer.write(payload.encode("utf-8"))
            await writer.drain()

            if writer.can_write_eof():
                writer.write_eof()

            data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            _LOGGER.debug(
                "Sophia NLU raw response (%d bytes): %s", len(data), data[:512]
            )
            text_resp = data.decode("utf-8").strip()

            results = []
            for line in text_resp.split("\n"):
                line = line.strip()
                if line:
                    results.append(json.loads(line))
            return results
        finally:
            writer.close()
            await writer.wait_closed()


    async def _async_send_to_nlu(
        self, text: str, conversation_id: str, language: str, is_reply: bool = False, device_id: str | None = None
    ) -> list[dict]:

        context: dict[str, Any] = {"is_reply": 1 if is_reply else 0}
        if device_id:
            context["device_id"] = device_id

        request = {
            "type": "recognize",
            "data": {
                "text": text,
                "language": language,
                "session_id": conversation_id,
                "context": context,
            },
        }

        try:
            results = await self._async_send_payload(request)
            return results
        except (OSError, asyncio.TimeoutError):
            raise
        except Exception as err:
            _LOGGER.error("Failed to send recognize command to Sophia NLU: %s", err)
            return []


    async def _async_send_response(
        self,
        session_id: str,
        intents: list[dict[str, Any]],
        text: str | None = None,
    ) -> str:
        """Send a 'response' request to the NLU server with per-intent results and return the output text.

        If text is provided (custom script / intent response), it is included as data.text so the NLU server can log it.
        """

        request: dict[str, Any] = {
            "type": "response",
            "data": {"context": {
                "session_id": session_id,
                "intents": intents,
            }}
        }
        if text:
            request["data"]["text"] = text

        try:
            results = await self._async_send_payload(request)
            if not results:
                return ""
            return results[0].get("data", {}).get("text", "")
        except (OSError, asyncio.TimeoutError):
            raise
        except Exception as err:
            _LOGGER.error("Failed to send response command to Sophia NLU: %s", err)
            return ""

    def _build_state_entry(self, state_obj: State) -> dict[str, Any]:
        """Build an enriched success entry dict from a HA State object."""
        attrs: dict[str, Any] = {}
        for k, v in state_obj.attributes.items():
            try:
                json.dumps(v)
                attrs[k] = v
            except (TypeError, ValueError):
                pass
        return {
            "id": state_obj.entity_id,
            "name": state_obj.name,
            "type": "entity",
            "state": state_obj.state,
            "attributes": attrs,
        }

    def _sanitize(self, value: Any) -> Any:
        """Recursively strip any non-JSON-serializable values from dicts/lists.

        - dicts: drop keys whose values cannot be serialized
        - lists/tuples: drop elements that cannot be serialized
        - scalars: return as-is if serializable, otherwise return None as sentinel
          (caller should discard the key)
        """
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for k, v in value.items():
                sanitized = self._sanitize(v)
                if sanitized is not _UNSERIALIZABLE:
                    result[k] = sanitized
            return result
        if isinstance(value, (list, tuple)):
            items = []
            for v in value:
                sanitized = self._sanitize(v)
                if sanitized is not _UNSERIALIZABLE:
                    items.append(sanitized)
            return items
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return _UNSERIALIZABLE

    def _normalize_intent_error(self, err: Exception) -> str:
        """Return a simplified error string for known noisy error types."""
        msg = str(err)
        if "MatchTargetsResult" in msg and "is_match=False" in msg:
            return "no_match"
        return msg

    def _is_reply(self, conversation_id: str) -> bool:
        """Return True if this turn should be flagged as a reply, ie. the previous turn for this conversation_id had continue_conversation=True."""
        session = self._conversation_sessions.get(conversation_id)
        return bool(session and session.get("continue_conversation"))

    def _update_session(
        self, conversation_id: str, continue_conversation: bool
    ) -> None:
        """Store/update per-session state after a turn completes."""
        self._conversation_sessions[conversation_id] = {
            "last_ts": time.monotonic(),
            "continue_conversation": continue_conversation,
        }

    async def check_instant_response(
        self,
        results: list[dict],
        text: str,
        conversation_id: str,
        language: str,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult | None:
        if len(results) != 1:
            return None

        intent_name = results[0].get("data", {}).get("intent", {}).get("name")
        if intent_name not in ("HassRespond", "HassClarification", "HassNevermind"):
            return None

        respond_text = results[0].get("data", {}).get("text", "").strip()
        respond_slots = {}

        # Get slots
        for entity in results[0].get("data", {}).get("entities", []):
            slot_name = entity.get("name", "")
            slot_value = entity.get("value", "")
            if slot_name:
                respond_slots[slot_name] = {"value": slot_value}

        try:
            respond_result = await intent.async_handle(
                self.hass,
                DOMAIN,
                "HassRespond",
                respond_slots,
                text,
                user_input.context,
                language=language,
                assistant=conversation.DOMAIN,
            )
        except Exception as err:
            _LOGGER.error("%s intent error: %s", intent_name, err)
            respond_result = intent.IntentResponse(language=language)

        # The server sends HassClarification / HassClarificationArea as a HassRespond, with continue_conversation flagged within the metadata
        metadata = results[0].get("data", {}).get("metadata") or {}
        continue_conversation = bool(metadata.get("continue_conversation")) or intent_name in ("HassClarification", "HassClarificationArea")

        respond_result.async_set_speech(respond_text or "Done.")
        return conversation.ConversationResult(
            response=respond_result,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )

    def _resolve_entities_from_slots(self, slots: dict[str, Any]) -> list[str]:
        """Resolve entity IDs from slots based on direct id, name/domain, area/floor names, or full domain lookup."""
        # 1. Direct entity ID match
        if "entity_id" in slots:
            entity_id = slots["entity_id"].get("value")
            if entity_id:
                return [entity_id]

        domain = slots.get("domain", {}).get("value")

        # 2. Match by friendly name + domain
        if "name" in slots and domain:
            name_slot = slots["name"].get("value", "").strip()
            if name_slot:
                name_lower = name_slot.lower()
                name_slug = name_lower.replace(" ", "_")
                for state_obj in self.hass.states.async_all(domain):
                    object_id = state_obj.entity_id.split(".", 1)[-1]
                    if state_obj.name.lower() == name_lower or object_id == name_slug or object_id.replace("_", " ") == name_lower:
                        return [state_obj.entity_id]
                _LOGGER.warning(
                    "No entity found in domain '%s' matching name '%s'",
                    domain,
                    name_slot,
                )
                return []

        # 3. Resolve by Floor or Area friendly names
        if "area" in slots or "floor" in slots:
            area_ids: set[str] = set()
            area_reg = ar.async_get(self.hass)

            # Match area by friendly name
            if "area" in slots:
                area_name = slots["area"].get("value", "").strip()
                if area_name:
                    area_entry = area_reg.async_get_area_by_name(area_name)
                    if area_entry:
                        area_ids.add(area_entry.id)
                    else:
                        _LOGGER.warning(
                            "Could not find registered area named '%s'", area_name
                        )

            # Match floor by friendly name, then gather all child area IDs
            elif "floor" in slots:
                floor_name = slots["floor"].get("value", "").strip()
                if floor_name:
                    floor_reg = fr.async_get(self.hass)
                    target_floor_id = None

                    # Find the floor ID matching the friendly name
                    for floor_entry in floor_reg.floors.values():
                        if floor_entry.name.lower() == floor_name.lower():
                            target_floor_id = floor_entry.floor_id
                            break

                    if target_floor_id:
                        # Grab all area IDs that belong to this floor ID
                        for area_entry in area_reg.areas.values():
                            if area_entry.floor_id == target_floor_id:
                                area_ids.add(area_entry.id)
                    else:
                        _LOGGER.warning(
                            "Could not find registered floor named '%s'", floor_name
                        )

            if not area_ids:
                return []

            # Gather matching entities within those resolved area IDs
            entity_reg = er.async_get(self.hass)
            device_reg = dr.async_get(self.hass)
            resolved_entities: list[str] = []

            for entity_entry in entity_reg.entities.values():
                # Filter by domain if it is specified in the intent slots
                if domain and entity_entry.domain != domain:
                    continue

                # Check if entity exposed
                if not self._is_entity_usable(entity_entry):
                    continue

                # Check if entity belongs to an area directly
                if entity_entry.area_id in area_ids:
                    resolved_entities.append(entity_entry.entity_id)
                    continue

                # Check if entity inherits its area via its physical device
                if entity_entry.device_id:
                    device_entry = device_reg.async_get(entity_entry.device_id)
                    if device_entry and device_entry.area_id in area_ids:
                        resolved_entities.append(entity_entry.entity_id)

            return resolved_entities

        # 4. Fallback: Grab all entities in the domain if no targeting criteria matched
        if domain:
            entity_reg = er.async_get(self.hass)
            return [
                entry.entity_id
                for entry in entity_reg.entities.values()
                if entry.domain == domain and self._is_entity_usable(entry)
            ]

        return []

    def _extract_service_payload(self, slots: dict[str, Any]) -> dict[str, Any]:
        """Filter out scoping fields and flatten slot values for core API delivery."""
        excluded_slots = {
            "id",
            "entity_id",
            "area",
            "floor",
            "name",
            "domain",
            "device_class",
            "state",
            "common_name",
        }
        payload = {}
        for key, slot_dict in slots.items():
            if key in excluded_slots:
                continue

            val = slot_dict.get("value")
            if val is None:
                continue

            # Brightness Handling
            if key == "brightness":
                if val == "down":
                    payload["brightness_step_pct"] = -20
                elif val == "up":
                    payload["brightness_step_pct"] = 20
                else:
                    try:
                        # Convert percentage or scale directly
                        payload["brightness_pct"] = int(val)
                    except ValueError:
                        pass

            # Color Name Handling
            elif key in ("color", "color_name"):
                payload["color_name"] = str(val).lower()

            # Color Temp Handling
            elif key == "color_temp":
                if val == "warm":
                    payload["color_temp_kelvin"] = 2700
                elif val == "cold" or val == "cool":
                    payload["color_temp_kelvin"] = 5000
                else:
                    try:
                        payload["color_temp_kelvin"] = int(val)
                    except ValueError:
                        pass

            # Position / Cover Handling (e.g., blinds, curtains, shades)
            elif key in ("position", "cover_position"):
                if val in ("open", "up"):
                    payload["position"] = 100
                elif val in ("close", "closed", "down"):
                    payload["position"] = 0
                else:
                    try:
                        payload["position"] = int(val)
                    except ValueError:
                        pass

            # Cover Relative Steps
            elif key == "position":
                if val in ("up", "open", "increase"):
                    payload["position"] = 100 
                elif val in ("down", "close", "decrease"):
                    payload["position"] = 0

            # Percentage Handling (e.g., fan speeds, generic percentages)
            elif key == "percentage":
                if val in ("up", "increase", "higher"):
                    # Standard fan or speed step
                    payload["percentage_step"] = 20
                elif val in ("down", "decrease", "lower"):
                    payload["percentage_step"] = -20
                else:
                    try:
                        payload["percentage"] = int(val)
                    except ValueError:
                        pass

            # Volume & Volume Steps (e.g., media players)
            elif key in ("volume", "volume_level"):
                try:
                    # The slot is always a 0-100 percentage; HA volume_level expects a float between 0.0 and 1.0
                    payload["volume_level"] = max(0.0, min(1.0, float(val) / 100.0))
                except ValueError:
                    pass

            elif key == "volume_step":
                if val in ("up", "increase", "louder"):
                    payload["volume_up"] = True  # Used directly with media_player.volume_up service
                elif val in ("down", "decrease", "softer", "quieter"):
                    payload["volume_down"] = True  # Used directly with media_player.volume_down service

            else:
                payload[key] = val

        return payload

    async def async_handle_standard_intent(
        self,
        intent_name: str,
        slots: dict[str, Any],
        user_input: conversation.ConversationInput,
    ) -> tuple[dict[str, Any], str | None]:
        text = user_input.text
        language = user_input.language or "en"

        intent_result: intent.IntentResponse | None = None
        intent_err_msg: str | None = None

        try:
            intent_result = await intent.async_handle(
                self.hass,
                DOMAIN,
                intent_name,
                slots,
                text,
                user_input.context,
                language=language,
                assistant=conversation.DOMAIN,
                device_id=user_input.device_id,
            )
        except intent.IntentHandleError as err:
            _LOGGER.error("Intent handling error for %s: %s", intent_name, err)
            intent_err_msg = self._normalize_intent_error(err)
        except intent.IntentUnexpectedError as err:
            _LOGGER.error("Unexpected intent error for %s: %s", intent_name, err)
            intent_err_msg = self._normalize_intent_error(err)
        except Exception as err:
            _LOGGER.exception("Failed to handle intent %s", intent_name)
            intent_err_msg = self._normalize_intent_error(err)

        # If HA errored and this isn't a custom-handled intent, emit an error entry
        if intent_result is None:
            intent_entry: dict[str, Any] = {
                "response_type": "error",
                "success": [],
                "failed": [],
                "error": intent_err_msg or "unknown error",
            }
            return intent_entry, None

        success_list = []
        for t in intent_result.success_results:  # type: ignore[union-attr]
            entry = dataclasses.asdict(t)
            if t.id and t.type == "entity":
                state_obj = self.hass.states.get(t.id)
                if state_obj is not None:
                    entry["state"] = state_obj.state
                    attrs: dict[str, Any] = {}
                    for k, v in state_obj.attributes.items():
                        try:
                            json.dumps(v)
                            attrs[k] = v
                        except (TypeError, ValueError):
                            pass
                    entry["attributes"] = attrs
            success_list.append(entry)

        # Build the per-intent context entry
        if intent_result is not None:
            response_type_val = intent_result.response_type.value
            failed_list = [dataclasses.asdict(t) for t in intent_result.failed_results]
            speech_slots_val = intent_result.speech_slots or None
            speech_text = intent_result.speech.get("plain", {}).get("speech", "") if intent_result.speech else ""
        else:
            response_type_val = "action_done"
            failed_list = []
            speech_slots_val = None
            speech_text = ""

        intent_entry = {
            "response_type": response_type_val,
            "success": success_list,
            "failed": failed_list,
        }
        if speech_slots_val:
            intent_entry["speech_slots"] = self._sanitize(speech_slots_val)
        if intent_err_msg:
            intent_entry["error"] = intent_err_msg

        return intent_entry, speech_text or None

    async def async_get_entity(
        self, entity_id: str, common_name: str | None
    ) -> tuple[bool, dict[str, Any]]:
        tmp_entity_name = " ".join(
            w
            for w in (
                word.rstrip("0123456789")
                for word in entity_id.split(".", 1)[-1].replace("_", " ").split()
            )
            if not w.isdigit() and w
        )

        if common_name is not None:
            tmp_entity_name = common_name

        state_obj = self.hass.states.get(entity_id)

        if state_obj is None:
            return False, {
                "id": entity_id,
                "name": tmp_entity_name,
                "type": "entity",
                "state": "unknown",
                "reason": "entity_not_found",
            }

        # 2. Sanitize and pack attributes safely for JSON encoding
        attrs: dict[str, Any] = {}
        for k, v in state_obj.attributes.items():
            try:
                json.dumps(v)
                attrs[k] = v
            except (TypeError, ValueError):
                pass

        entity_name = (
            state_obj.attributes.get("friendly_name")
            or state_obj.name
            or tmp_entity_name
        )

        entity_payload = {
            "id": state_obj.entity_id,
            "name": entity_name,
            "type": "entity",
            "state": state_obj.state,
            "attributes": attrs,
        }

        # 3. Determine Success vs Failure based on operational state
        # If a device is offline or dead, count it as a failure state
        if state_obj.state in ("unavailable", "unknown"):
            entity_payload["reason"] = "device_offline"
            return False, entity_payload

        return True, entity_payload

    def _pick_tomorrow_forecast(self, forecast_list: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the daily forecast entry for tomorrow, if available."""
        today = dt_util.now().date()
        dated: list[tuple[date, dict[str, Any]]] = []
        for entry in forecast_list:
            raw = entry.get("datetime")
            parsed = dt_util.parse_datetime(str(raw)) if raw is not None else None
            if parsed is None:
                continue
            dated.append((dt_util.as_local(parsed).date(), entry))
        future = [item for item in dated if item[0] > today]
        if future:
            future.sort(key=lambda item: item[0])
            return future[0][1]
        if len(forecast_list) > 1:
            return forecast_list[1]
        return forecast_list[0] if forecast_list else None

    async def _async_apply_tomorrow_forecast(self, entity_id: str, entity: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Replace current weather on the entity payload with tomorrow's daily forecast."""
        try:
            result = await self.hass.services.async_call("weather", "get_forecasts", {"type": "daily", "entity_id": entity_id}, blocking=True, return_response=True)
        except Exception as err:
            _LOGGER.error("Failed to get tomorrow forecast for %s: %s", entity_id, err)
            entity["reason"] = "forecast_unavailable"
            return False, entity

        forecast_list: list[dict[str, Any]] = []
        if isinstance(result, dict):
            forecast_list = (result.get(entity_id) or {}).get("forecast") or []
        tomorrow = self._pick_tomorrow_forecast(forecast_list)
        if not tomorrow:
            entity["reason"] = "forecast_unavailable"
            return False, entity

        if condition := tomorrow.get("condition"):
            entity["state"] = condition
        attrs = entity.setdefault("attributes", {})
        for key, val in tomorrow.items():
            if key == "condition" or val is None:
                continue
            try:
                json.dumps(val)
            except (TypeError, ValueError):
                continue
            attrs[key] = val
        return True, entity

    def _speech_from_payload(self, payload: Any) -> str | None:
        """Return the first speech-like string found in a service response.

        Responses can be a plain string, a dict with a speech/text/message/
        response key, or a dict keyed by entity_id wrapping such a value.
        """
        if isinstance(payload, str):
            return payload.strip() or None
        if not isinstance(payload, dict):
            return None
        for key in ("speech", "text", "message", "response"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for v in payload.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
            nested = self._speech_from_payload(v)
            if nested:
                return nested
        return None

    def _render_static_value(self, val: Any, variables: dict[str, Any] | None = None) -> str | None:
        """Render a static config value that may be a string or Template."""
        if isinstance(val, str):
            return val.strip() or None
        if hasattr(val, "async_render"):
            try:
                rendered = val.async_render(variables or {}, parse_result=False)
                return str(rendered).strip() or None
            except Exception:
                return None
        return None

    def _find_conversation_response(
        self, actions: list, variables: dict[str, Any] | None = None
    ) -> str | None:
        speech = None
        for action in actions:
            if not isinstance(action, dict):
                continue
            if "set_conversation_response" in action:
                val = action.get("set_conversation_response")
                if val is None:
                    speech = None
                else:
                    rendered = self._render_static_value(val, variables)
                    if rendered is not None:
                        speech = rendered
            for key in ("sequence", "then", "else", "default"):
                nested = action.get(key)
                if isinstance(nested, list):
                    found = self._find_conversation_response(nested, variables)
                    if found is not None:
                        speech = found
            choose = action.get("choose")
            if isinstance(choose, list):
                for choice in choose:
                    if isinstance(choice, dict) and isinstance(choice.get("sequence"), list):
                        found = self._find_conversation_response(choice["sequence"], variables)
                        if found is not None:
                            speech = found
            repeat = action.get("repeat")
            if isinstance(repeat, dict) and isinstance(repeat.get("sequence"), list):
                found = self._find_conversation_response(repeat["sequence"], variables)
                if found is not None:
                    speech = found
            if_action = action.get("if")
            if isinstance(if_action, list):
                for branch in if_action:
                    if isinstance(branch, dict):
                        for key in ("then", "else"):
                            nested = branch.get(key)
                            if isinstance(nested, list):
                                found = self._find_conversation_response(nested, variables)
                                if found is not None:
                                    speech = found
            parallel = action.get("parallel")
            if isinstance(parallel, list):
                for branch in parallel:
                    if isinstance(branch, dict):
                        nested = branch.get("sequence")
                        if isinstance(nested, list):
                            found = self._find_conversation_response(nested, variables)
                            if found is not None:
                                speech = found
                    elif isinstance(branch, list):
                        found = self._find_conversation_response(branch, variables)
                        if found is not None:
                            speech = found
        return speech

    def _static_script_speech(
        self, entity_id: str, variables: dict[str, Any] | None = None
    ) -> str | None:
        component = self.hass.data.get("script")
        if component is None or not hasattr(component, "get_entity"):
            return None
        script_entity = component.get_entity(entity_id)
        raw = getattr(script_entity, "raw_config", None) if script_entity is not None else None
        if not isinstance(raw, dict):
            return None
        sequence = raw.get("sequence") or []
        if not isinstance(sequence, list):
            return None
        return self._find_conversation_response(sequence, variables)

    async def _async_run_script(self, entity_id: str, variables: dict[str, Any], context: Any) -> str | None:
        """Run a script entity once and return its conversation speech, if any.

        Runs via the script entity's Script object directly so both
        conversation_response (from set_conversation_response) and
        service_response (from response_variable, even non-dict values) are
        captured without HA's dict-only service validation or double execution.
        Falls back to service calls for mocked environments without entities.
        """
        component = self.hass.data.get("script")
        script_entity = None
        if component is not None and hasattr(component, "get_entity"):
            try:
                script_entity = component.get_entity(entity_id)
            except Exception:
                script_entity = None

        script_obj = getattr(script_entity, "script", None) if script_entity is not None else None
        if script_obj is not None and hasattr(script_obj, "async_run"):
            result = await script_obj.async_run(variables or {}, context)
            if result is not None:
                conv = getattr(result, "conversation_response", None)
                if isinstance(conv, str) and conv.strip():
                    return conv.strip()
                svc = getattr(result, "service_response", None)
                speech = self._speech_from_payload(svc)
                if speech:
                    return speech
            return self._static_script_speech(entity_id, variables)

        object_id = entity_id.split(".", 1)[-1]
        if self.hass.services.has_service("script", object_id):
            try:
                result = await self.hass.services.async_call(
                    "script", object_id, variables, blocking=True,
                    context=context, return_response=True,
                )
            except Exception:
                result = None
            return self._speech_from_payload(result) or self._static_script_speech(entity_id, variables)

        turn_on_data: dict[str, Any] = {"entity_id": entity_id}
        if variables:
            turn_on_data["variables"] = variables
        await self.hass.services.async_call(
            "script", "turn_on", turn_on_data, blocking=True, context=context
        )
        return self._static_script_speech(entity_id, variables)

    async def _async_try_script_by_name(
        self, intent_name: str, slots: dict[str, Any], context: Any
    ) -> tuple[bool, str | None, str | None]:
        """Run a script whose entity_id or friendly name matches a custom intent.

        Returns (handled, speech, entity_id). Used when no intent_script
        handler exists for the custom intent name.
        """
        candidates: list[str] = []
        normalized = intent_name.strip().lower().replace(" ", "_").replace("-", "_")
        normalized = "".join(c for c in normalized if c.isalnum() or c == "_")
        if normalized:
            candidates.append(f"script.{normalized}")

        for state_obj in self.hass.states.async_all("script"):
            name_lower = state_obj.name.lower()
            intent_lower = intent_name.strip().lower()
            object_id = state_obj.entity_id.split(".", 1)[-1]
            if (
                name_lower == intent_lower
                or object_id == normalized
                or object_id.replace("_", " ") == intent_lower
            ):
                if state_obj.entity_id not in candidates:
                    candidates.append(state_obj.entity_id)

        for entity_id in candidates:
            exists = (
                self.hass.states.get(entity_id) is not None
                or self.hass.services.has_service("script", entity_id.split(".", 1)[-1])
            )
            if not exists:
                component = self.hass.data.get("script")
                if component is not None and hasattr(component, "get_entity"):
                    try:
                        exists = component.get_entity(entity_id) is not None
                    except Exception:
                        exists = False
            if not exists:
                continue
            service_data = self._extract_service_payload(slots)
            speech = await self._async_run_script(entity_id, service_data, context)
            return True, speech, entity_id
        return False, None, None

    async def _async_handle_alarm_intent(self, intent_name: str, slots: dict[str, Any], common_name: str | None, context: Any) -> dict[str, Any]:
        """Execute an alarm intent via alarm_control_panel services and return the per-intent context entry.

        Validation failures (missing entity or bad mode) produce an error entry without any service call; speech is generated by the NLU engine from the entry.
        """
        intent_entry: dict[str, Any] = {"response_type": "action_done", "success": [], "failed": []}

        entity_id = str(slots.get("entity_id", {}).get("value") or "").strip()
        if not entity_id:
            _LOGGER.error("%s received without an entity_id slot", intent_name)
            intent_entry.update({"response_type": "error", "error": "missing_entity_id"})
            return intent_entry

        # Determine service
        if intent_name == "HassAlarmSetMode":
            raw_mode = str(slots.get("mode", {}).get("value") or "").strip()
            mode = raw_mode.lower().replace(" ", "_").replace("-", "_")
            service_name = ALARM_MODE_SERVICES.get(mode)
            if service_name is None:
                if not raw_mode:
                    _LOGGER.error("HassAlarmSetMode received without a mode slot for %s", entity_id)
                    intent_entry.update({"response_type": "error", "error": "missing_alarm_mode"})
                    return intent_entry
                _LOGGER.error("HassAlarmSetMode received unrecognized mode '%s' for %s", raw_mode, entity_id)
                intent_entry.update({"response_type": "error", "error": "invalid_alarm_mode", "mode": raw_mode})
                return intent_entry
        else:
            service_name = ALARM_FIXED_SERVICES[intent_name]

        # Only pass code if provided and the entity actually accepts one
        service_data: dict[str, Any] = {"entity_id": entity_id}
        code = slots.get("code", {}).get("value")
        state_obj = self.hass.states.get(entity_id)
        if code is not None and str(code).strip() and state_obj is not None and state_obj.attributes.get("code_format") is not None:
            service_data["code"] = str(code).strip()

        try:
            await self.hass.services.async_call(ALARM_DOMAIN, service_name, service_data, blocking=True, context=context)
            _LOGGER.debug("Fired alarm service %s.%s for %s (code provided: %s)", ALARM_DOMAIN, service_name, entity_id, "code" in service_data)
        except Exception as err:
            _LOGGER.error("Alarm service %s.%s failed for %s: %s", ALARM_DOMAIN, service_name, entity_id, err)
            intent_entry.update({"response_type": "error", "error": self._normalize_intent_error(err)})
            return intent_entry

        is_success, entity = await self.async_get_entity(entity_id, common_name)
        if is_success:
            intent_entry["success"].append(entity)
        else:
            intent_entry["failed"].append(entity)
        return intent_entry

    def _entity_area_id(self, entity_id: str) -> str | None:
        """Return the area an entity is in, directly or via its device."""
        entry = er.async_get(self.hass).async_get(entity_id)
        if entry is None:
            return None
        if entry.area_id:
            return entry.area_id
        if entry.device_id:
            device = dr.async_get(self.hass).async_get(entry.device_id)
            return device.area_id if device else None
        return None

    def _find_capable_entities(self, slots: dict[str, Any], domain: str, features: int, ignore_location: bool) -> list[str]:
        """Resolve entities for the slots in the given domain that support all the given features and are not unavailable.

        ignore_location drops the area / floor slots, for intents where those are service parameters rather than a place to look for the device.
        """
        resolve_slots = {k: v for k, v in slots.items() if not (ignore_location and k in ("area", "floor"))}
        resolve_slots["domain"] = {"value": domain}
        result: list[str] = []
        for entity_id in self._resolve_entities_from_slots(resolve_slots):
            state_obj = self.hass.states.get(entity_id)
            if state_obj is None or entity_id.split(".", 1)[0] != domain or state_obj.state in ("unavailable", "unknown"):
                continue
            if (int(state_obj.attributes.get("supported_features") or 0) & features) != features:
                continue
            result.append(entity_id)
        return result

    async def _async_finish_entities(self, intent_entry: dict[str, Any], entity_ids: list[str], common_name: str | None, extra: dict[str, Any] | None = None) -> None:
        """Read back the entities after a service call and sort them into the success / failed lists."""
        for entity_id in entity_ids:
            is_success, entity = await self.async_get_entity(entity_id, common_name if len(entity_ids) == 1 else None)
            if extra:
                entity.update(extra)
            intent_entry["success" if is_success else "failed"].append(entity)

    async def _async_handle_vacuum_clean_area(self, slots: dict[str, Any], context: Any) -> dict[str, Any]:
        """Execute HassVacuumCleanArea via vacuum.clean_area.  The area slot is the area to clean (cleaning_area_id), not where the vacuum lives."""
        intent_entry: dict[str, Any] = {"response_type": "action_done", "success": [], "failed": []}

        area_name = str(slots.get("area", {}).get("value") or "").strip()
        if not area_name:
            _LOGGER.error("HassVacuumCleanArea received without an area slot")
            intent_entry.update({"response_type": "error", "error": "missing_area"})
            return intent_entry

        areas = list(intent.find_areas(area_name, ar.async_get(self.hass)))
        if not areas:
            _LOGGER.error("HassVacuumCleanArea: no area named '%s'", area_name)
            intent_entry.update({"response_type": "error", "error": "invalid_area"})
            return intent_entry
        area_ids = [a.id for a in areas]

        vacuums = self._find_capable_entities(slots, VACUUM_DOMAIN, VACUUM_CLEAN_AREA_FEATURE, ignore_location=True)
        if len(vacuums) > 1 and "name" not in slots and "entity_id" not in slots:
            # No specific vacuum requested, so prefer the ones in the area being cleaned
            in_area = [v for v in vacuums if self._entity_area_id(v) in area_ids]
            vacuums = in_area or vacuums
        if not vacuums:
            _LOGGER.error("HassVacuumCleanArea: no vacuum supporting clean_area found")
            intent_entry.update({"response_type": "error", "error": "no_match"})
            return intent_entry

        try:
            await self.hass.services.async_call(VACUUM_DOMAIN, "clean_area", {"entity_id": vacuums, "cleaning_area_id": area_ids}, blocking=True, context=context)
            _LOGGER.debug("Fired vacuum.clean_area for %s areas %s", vacuums, area_ids)
        except Exception as err:
            _LOGGER.error("vacuum.clean_area failed for %s: %s", vacuums, err)
            intent_entry.update({"response_type": "error", "error": self._normalize_intent_error(err)})
            return intent_entry

        await self._async_finish_entities(intent_entry, vacuums, slots.get("common_name", {}).get("value"))
        return intent_entry

    @staticmethod
    def _media_field(item: Any, key: str) -> Any:
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

    async def _async_handle_media_search_and_play(self, slots: dict[str, Any], context: Any) -> dict[str, Any]:
        """Execute HassMediaSearchAndPlay: media_player.search_media, then media_player.play_media with the first result."""
        intent_entry: dict[str, Any] = {"response_type": "action_done", "success": [], "failed": []}

        search_query = str(slots.get("search_query", {}).get("value") or "").strip()
        if not search_query:
            _LOGGER.error("HassMediaSearchAndPlay received without a search_query slot")
            intent_entry.update({"response_type": "error", "error": "missing_search_query"})
            return intent_entry

        players = self._find_capable_entities(slots, MEDIA_PLAYER_DOMAIN, MEDIA_SEARCH_FEATURES, ignore_location=False)
        if len(players) > 1:
            # Prefer a player that is actually on, and use a single one like HA does
            players = [p for p in players if self.hass.states.get(p).state != "off"] or players
        if not players:
            _LOGGER.error("HassMediaSearchAndPlay: no media player supporting search and play found")
            intent_entry.update({"response_type": "error", "error": "no_match"})
            return intent_entry
        entity_id = players[0]

        search_data: dict[str, Any] = {"entity_id": entity_id, "search_query": search_query}
        media_class = slots.get("media_class", {}).get("value")
        if media_class:
            search_data["media_filter_classes"] = [media_class]

        try:
            search_response = await self.hass.services.async_call(MEDIA_PLAYER_DOMAIN, "search_media", search_data, blocking=True, context=context, return_response=True)
        except Exception as err:
            _LOGGER.error("media_player.search_media failed for %s: %s", entity_id, err)
            intent_entry.update({"response_type": "error", "error": self._normalize_intent_error(err)})
            return intent_entry

        entity_response = (search_response or {}).get(entity_id)
        results = self._media_field(entity_response, "result") if entity_response is not None else None
        if not results:
            _LOGGER.debug("HassMediaSearchAndPlay: no results for '%s' on %s", search_query, entity_id)
            is_success, entity = await self.async_get_entity(entity_id, slots.get("common_name", {}).get("value"))
            entity["reason"] = "media_not_found"
            intent_entry["failed"].append(entity)
            return intent_entry

        first = results[0]
        content_id = self._media_field(first, "media_content_id")
        content_type = self._media_field(first, "media_content_type")
        try:
            await self.hass.services.async_call(MEDIA_PLAYER_DOMAIN, "play_media", {"entity_id": entity_id, "media_content_id": content_id, "media_content_type": content_type}, blocking=True, context=context)
            _LOGGER.debug("Fired media_player.play_media for %s with %s (%s)", entity_id, content_id, content_type)
        except Exception as err:
            _LOGGER.error("media_player.play_media failed for %s: %s", entity_id, err)
            intent_entry.update({"response_type": "error", "error": self._normalize_intent_error(err)})
            return intent_entry

        media_info = {"title": self._media_field(first, "title"), "media_class": str(self._media_field(first, "media_class") or ""), "media_content_type": str(content_type or "")}
        await self._async_finish_entities(intent_entry, [entity_id], slots.get("common_name", {}).get("value"), {"media": media_info})
        return intent_entry

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        text = user_input.text.strip()
        conversation_id = user_input.conversation_id or ulid.ulid()
        language = user_input.language or "en"
        response = intent.IntentResponse(language=language)

        if not text:
            response.async_set_speech("No text provided.")
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )

        is_reply = self._is_reply(conversation_id)
        try:
            results = await self._async_send_to_nlu(
                text, conversation_id, language, is_reply, user_input.device_id
            )
        except asyncio.TimeoutError:
            _LOGGER.error("Sophia NLU at %s:%s did not respond in time", self._host, self._port)
            response.async_set_speech("Sorry, the NLU engine took too long to respond.")
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )
        except OSError as err:
            _LOGGER.error(
                "Failed to connect to Sophia NLU at %s:%s: %s",
                self._host,
                self._port,
                err,
            )
            response.async_set_speech("Sorry, I could not connect to the NLU engine.")
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )
        except Exception as err:
            _LOGGER.exception("Error communicating with Sophia NLU")
            response.async_set_speech(f"Error communicating with NLU engine: {err}")
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )

        if not results:
            response.async_set_speech("Sorry, I didn't understand that.")
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )

        # Immediately provide response if HassRespond or HassClarification
        instant_response = await self.check_instant_response(
            results, text, conversation_id, language, user_input
        )
        if instant_response is not None:
            self._update_session(
                conversation_id, instant_response.continue_conversation
            )
            return instant_response

        # Process each intent returned by the NLU engine.
        # The server may return multiple JSONL lines, one per detected intent.
        last_intent_result: intent.IntentResponse | None = None
        session_id: str | None = None
        intents_context: list[dict[str, Any]] = []
        script_response_text: str | None = None
        custom_intent_speech: str | None = None

        for result in results:
            # Wyoming error response
            resp_type = result.get("type", "")
            if resp_type == "error":
                error_data = result.get("data", {})
                error_text = error_data.get("text", "Unknown error from NLU engine")
                _LOGGER.error(
                    "Sophia NLU returned error: %s (code=%s)",
                    error_text,
                    error_data.get("code"),
                )
                continue

            # Wyoming intent response
            data = result.get("data", {})
            intent_info = data.get("intent", {})
            intent_name = intent_info.get("name", "")

            if not intent_name:
                _LOGGER.warning(
                    "Sophia NLU returned a result with no intent name, skipping"
                )
                continue

            # Grab session_id from the first valid result
            if session_id is None:
                session_id = data.get("metadata", {}).get("session_id")

            # Map Wyoming entities to HA intent slots: [{"name": "x", "value": "y"}] -> {"x": {"value": "y"}}
            slots = {}
            for entity in data.get("entities", []):
                slot_name = entity.get("name", "")
                slot_value = entity.get("value", "")
                if slot_name:
                    slots[slot_name] = {"value": slot_value}

            if "common_name" in slots:
                common_name = slots["common_name"].get("value")
            else:
                common_name = None

            # Get entity IDs
            devices = self._resolve_entities_from_slots(slots)
            if len(devices) != 1:
                common_name = None

            # Alarm intents map directly to alarm_control_panel services
            if intent_name in ALARM_INTENTS:
                intents_context.append(await self._async_handle_alarm_intent(intent_name, slots, common_name, user_input.context))
                if last_intent_result is None:
                    last_intent_result = intent.IntentResponse(language=language)
                    last_intent_result.response_type = intent.IntentResponseType.ACTION_DONE
                continue

            # Vacuum area cleaning and media search need multi-step handling
            if intent_name in ("HassVacuumCleanArea", "HassMediaSearchAndPlay"):
                handler = self._async_handle_vacuum_clean_area if intent_name == "HassVacuumCleanArea" else self._async_handle_media_search_and_play
                intents_context.append(await handler(slots, user_input.context))
                if last_intent_result is None:
                    last_intent_result = intent.IntentResponse(language=language)
                    last_intent_result.response_type = intent.IntentResponseType.ACTION_DONE
                continue

            if not is_standard_intent(intent_name):
                # Custom intents with an intent_script handler fall through to
                # standard HA handling below (ScriptIntentHandler executes the
                # action and renders speech). Only when no handler exists, try
                # a script with a matching entity_id or friendly name.
                has_handler = isinstance(
                    (self.hass.data.get("intent_script") or {}).get(intent_name), dict
                )
                if not has_handler:
                    try:
                        handled, speech, script_entity = await self._async_try_script_by_name(
                            intent_name, slots, user_input.context
                        )
                    except Exception as err:
                        _LOGGER.error("Script fallback for %s failed: %s", intent_name, err)
                        handled, speech, script_entity = False, None, None
                    if handled and script_entity:
                        is_success, script_state = await self.async_get_entity(
                            script_entity, common_name
                        )
                        script_entry: dict[str, Any] = {
                            "response_type": "action_done",
                            "success": [script_state] if is_success else [],
                            "failed": [] if is_success else [script_state],
                        }
                        intents_context.append(script_entry)
                        if speech:
                            script_response_text = speech
                        if last_intent_result is None:
                            last_intent_result = intent.IntentResponse(language=language)
                            last_intent_result.response_type = intent.IntentResponseType.ACTION_DONE
                        continue

            # Standard HA intent system
            if (
                len(devices) == 0
                or "Timer" in intent_name
                or "List" in intent_name
                or intent_name in ["HassBroadcast", "HassNevermind"]
            ):
                intent_entry, speech_text = await self.async_handle_standard_intent(
                    intent_name, slots, user_input
                )
                intents_context.append(intent_entry)

                if not is_standard_intent(intent_name) and speech_text:
                    custom_intent_speech = speech_text

                if last_intent_result is None:
                    last_intent_result = intent.IntentResponse(language=language)
                    last_intent_result.response_type = (
                        intent.IntentResponseType.ACTION_DONE
                    )
                continue

            entity_domain = devices[0].split(".")[0]
            service_target = get_service_call(intent_name, entity_domain)

            # Fallback to standard HA intent handling if no service mapping exists
            if service_target is None:
                _LOGGER.debug("No service mapping for %s + %s, falling back to standard intent handling", intent_name, entity_domain)
                # HA matches the name slot against the full entity name (device name + entity name), which the NLU engine's short name may not equal; HA also accepts an entity_id as the name, so pass the resolved one.
                if "name" in slots:
                    slots["name"] = {"value": devices[0]}
                intent_entry, speech_text = await self.async_handle_standard_intent(
                    intent_name, slots, user_input
                )
                # Tomorrow's forecast is not part of the weather entity state, so replace the current conditions with a weather.get_forecasts service call
                if intent_name == "HassGetWeather" and slots.get("time", {}).get("value") == "tomorrow":
                    forecast_success: list[dict[str, Any]] = []
                    for entry in intent_entry.get("success", []):
                        is_success, entry = await self._async_apply_tomorrow_forecast(entry.get("id") or devices[0], entry)
                        (forecast_success if is_success else intent_entry.setdefault("failed", [])).append(entry)
                    intent_entry["success"] = forecast_success
                intents_context.append(intent_entry)

                # Capture speech for custom intents
                if not is_standard_intent(intent_name) and speech_text:
                    custom_intent_speech = speech_text

                if last_intent_result is None:
                    last_intent_result = intent.IntentResponse(language=language)
                    last_intent_result.response_type = (
                        intent.IntentResponseType.ACTION_DONE
                    )
                continue

            intent_entry: dict[str, Any] = {
                "response_type": "query_answer",
                "success": [],
                "failed": [],
            }

            # GO through devices
            for entity_id in devices:
                service_data = self._extract_service_payload(slots)
                settle_watch: _SettleWatch | None = None

                try:
                    if entity_id.startswith("script."):
                        speech = await self._async_run_script(entity_id, service_data, user_input.context)
                        _LOGGER.debug("Executed script %s with %s", entity_id, service_data)
                        intent_entry["response_type"] = "action_done"
                        if speech:
                            script_response_text = speech
                    else:
                        service_domain, service_name = service_target
                        service_data["entity_id"] = entity_id
                        if intent_name in ("HassMediaPlayerMute", "HassMediaPlayerUnmute"):
                            service_data["is_volume_muted"] = intent_name == "HassMediaPlayerMute"

                        # Start watching before the call so a fast state update isn't missed
                        settle_predicate = _build_settle_predicate(service_domain, service_data, self.hass.states.get(entity_id))
                        if settle_predicate is not None:
                            settle_watch = _SettleWatch(self.hass, entity_id, settle_predicate)

                        await self.hass.services.async_call(service_domain, service_name, service_data, blocking=True)
                        _LOGGER.debug("Bypassed intent pipeline. Fired service %s.%s with %s", service_domain, service_name, service_data)
                        intent_entry["response_type"] = "action_done"

                        # Wait (up to SETTLE_TIMEOUT) for the device to report its new state; returns instantly if it already has
                        if settle_watch is not None:
                            await settle_watch.wait()

                except Exception as err:
                    _LOGGER.error("Direct execution fallback failed for %s: %s", entity_id, err)
                    intent_entry["response_type"] = "error"
                    continue
                finally:
                    if settle_watch is not None:
                        settle_watch.cancel()

                # Get entity
                is_success, entity = await self.async_get_entity(entity_id, common_name)
                if is_success:
                    intent_entry["success"].append(entity)
                else:
                    intent_entry["failed"].append(entity)
            intents_context.append(intent_entry)

            if last_intent_result is None:
                last_intent_result = intent.IntentResponse(language=language)
                last_intent_result.response_type = intent.IntentResponseType.ACTION_DONE

        # Send per-intent results back to the NLU engine to generate output text
        # Priority: script response > custom intent response > NLU-generated response
        # The response request is always sent (for logging), with the custom response text included if one exists.
        custom_speech = script_response_text or custom_intent_speech
        nlu_speech = ""
        if session_id:
            try:
                nlu_speech = await self._async_send_response(session_id, intents_context, custom_speech)
            except (OSError, asyncio.TimeoutError) as err:
                _LOGGER.error("Failed to get response text from Sophia NLU: %s", err)
            except Exception as err:
                _LOGGER.exception("Error getting response text from Sophia NLU")
        else:
            _LOGGER.warning("No session_id in NLU response; cannot generate output text")
        final_speech = custom_speech or nlu_speech

        # Finish up: set the output text and return the conversation result to HA
        if last_intent_result is not None:
            last_intent_result.async_set_speech(final_speech or "Done.")
            self._update_session(conversation_id, False)
            return conversation.ConversationResult(
                response=last_intent_result, conversation_id=conversation_id
            )

        response.async_set_speech(final_speech or "Sorry, I didn't understand that.")
        self._update_session(conversation_id, False)
        return conversation.ConversationResult(
            response=response, conversation_id=conversation_id
        )
