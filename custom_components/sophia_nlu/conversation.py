"""The conversation platform for the Sophia NLU integration."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    device_registry as dr,
    floor_registry as fr,
    intent,
)


from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import ulid

from .const import CONF_HOST, CONF_PORT, DOMAIN, get_service_call

_LOGGER = logging.getLogger(__name__)

# Sentinel used by _sanitize() to signal a value cannot be JSON-serialized
_UNSERIALIZABLE = object()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities, emulating home if yaml_file is configured."""
    yaml_file = hass.data.get(DOMAIN, {}).get("yaml_file")

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

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._host = entry.data[CONF_HOST]
        self._port = entry.data[CONF_PORT]
        self.emulated_home = None

        self._attr_unique_id = entry.entry_id
        self.entity_id = f"conversation.{entry.domain}"
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
        self, text: str, conversation_id: str, language: str, is_reply: bool = False
    ) -> list[dict]:

        request = {
            "type": "recognize",
            "data": {
                "text": text,
                "language": language,
                "session_id": conversation_id,
                "context": {"is_reply": 1 if is_reply else 0},
            },
        }

        try:
            results = await self._async_send_payload(request)
            return results
        except Exception as err:
            _LOGGER.error(f"Failed to send recognize command to Sophia NLU: {err}")


    async def _async_send_response(
        self,
        session_id: str,
        intents: list[dict[str, Any]],
    ) -> str:
        """Send a 'response' request to the NLU server with per-intent results and return the output text."""

        request = {
            "type": "response",
            "data": {"context": {
                "session_id": session_id,
                "intents": intents,
            }}
        }

        try:
            results = await self._async_send_payload(request)
            return results[0].get("data", {}).get("text", "")
        except Exception as err:
            _LOGGER.error(f"Failed to send response command to Sophia NLU: {err}")

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
        """Return True if this turn should be flagged as a reply.

        True if the previous turn had continue_conversation=True, or if the
        previous turn for this conversation_id was received within the last 30 seconds.
        """
        session = self._conversation_sessions.get(conversation_id)
        if session is None:
            return False
        if session.get("continue_conversation"):
            return True
        last_ts = session.get("last_ts")
        if last_ts is not None and (time.monotonic() - last_ts) <= 30.0:
            return True
        return False

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

        respond_result.async_set_speech(respond_text or "Done.")
        return conversation.ConversationResult(
            response=respond_result,
            conversation_id=conversation_id,
            continue_conversation=(intent_name == "HassClarification"),
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
                for state_obj in self.hass.states.async_all(domain):
                    if state_obj.name.lower() == name_slot.lower():
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
            return [
                state_obj.entity_id for state_obj in self.hass.states.async_all(domain)
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
            if key not in excluded_slots:
                val = slot_dict.get("value")
                if val is not None:
                    payload[key] = val
        return payload

    async def async_handle_standard_intent(
        self,
        intent_name: str,
        slots: dict[str, Any],
        user_input: conversation.ConversationInput,
    ) -> dict[str, Any]:
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
            return intent_entry

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
        else:
            response_type_val = "action_done"
            failed_list = []
            speech_slots_val = None

        intent_entry = {
            "response_type": response_type_val,
            "success": success_list,
            "failed": failed_list,
        }
        if speech_slots_val:
            intent_entry["speech_slots"] = self._sanitize(speech_slots_val)
        if intent_err_msg:
            intent_entry["error"] = intent_err_msg

        return intent_entry

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
                text, conversation_id, language, is_reply
            )
        except (OSError, asyncio.TimeoutError) as err:
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

            # Standard HA intent system
            if (
                len(devices) == 0
                or "Timer" in intent_name
                or "List" in intent_name
                or intent_name in ["HassBroadcast", "HassNevermind"]
            ):
                intent_entry = await self.async_handle_standard_intent(
                    intent_name, slots, user_input
                )
                intents_context.append(intent_entry)

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

            entity_domain = devices[0].split(".")[0]
            service_target = get_service_call(intent_name, entity_domain)

            # GO through devices
            for entity_id in devices:
                if service_target is not None:
                    service_domain, service_name = service_target
                    service_data = self._extract_service_payload(slots)
                    service_data["entity_id"] = entity_id

                    try:
                        await self.hass.services.async_call(
                            service_domain,
                            service_name,
                            service_data,
                            blocking=True,
                        )
                        _LOGGER.debug(
                            "Bypassed intent pipeline. Fired service %s.%s with %s",
                            service_domain,
                            service_name,
                            service_data,
                        )
                        intent_entry["response_type"] = "action_done"
                    except Exception as err:
                        _LOGGER.error(
                            "Direct execution fallback failed for %s: %s",
                            entity_id,
                            err,
                        )
                        intent_entry["response_type"] = "error"
                        continue

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
        if session_id:
            try:
                final_speech = await self._async_send_response(
                    session_id,
                    intents_context,
                )
            except (OSError, asyncio.TimeoutError) as err:
                _LOGGER.error("Failed to get response text from Sophia NLU: %s", err)
                final_speech = ""
            except Exception as err:
                _LOGGER.exception("Error getting response text from Sophia NLU")
                final_speech = ""
        else:
            _LOGGER.warning(
                "No session_id in NLU response; cannot generate output text"
            )
            final_speech = ""

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
