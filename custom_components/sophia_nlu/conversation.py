"""The conversation platform for the Sophia NLU integration."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar, entity_registry as er, intent
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import ulid

from .const import CONF_HOST, CONF_PORT, DOMAIN

_LOGGER = logging.getLogger(__name__)

# Sentinel used by _sanitize() to signal a value cannot be JSON-serialized
_UNSERIALIZABLE = object()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    async_add_entities([SophiaNLUConversationEntity(config_entry)])


class SophiaNLUConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Sophia NLU conversation agent entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._host = entry.data[CONF_HOST]
        self._port = entry.data[CONF_PORT]
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self.entry.entry_id)},
            name="Sophia NLU Engine",
            manufacturer="Aquila Labs",
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return ["en"]

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_send_to_nlu(
        self, text: str, conversation_id: str, language: str
    ) -> list[dict]:
        """Send a Wyoming 'recognize' request to the Sophia NLU server and return parsed JSON responses.

        The server may return multiple newline-separated JSON objects (JSONL)
        when multiple intents are detected.  Each non-empty line is parsed and
        returned as a separate dict in the list.
        """
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port), timeout=5.0
        )
        try:
            request = json.dumps(
                {
                    "type": "recognize",
                    "data": {
                        "text": text,
                        "language": language,
                        "session_id": conversation_id,
                    },
                },
                separators=(",", ":"),
            )
            payload = f"{request}\n"
            # payload = f"{len(request)}\n{request}\n"
            writer.write(payload.encode("utf-8"))
            await writer.drain()

            # Signal we are done sending so the server sees EOF on its read half.
            # This also prompts the server task to finish and close the connection,
            # allowing our read() below to complete instead of hanging.
            if writer.can_write_eof():
                writer.write_eof()

            # Server responds with raw JSON (no length prefix).  Read all data until the
            # server closes the connection, then parse each line as a separate JSON object.
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

    async def _async_send_response(
        self,
        session_id: str,
        intents: list[dict[str, Any]],
    ) -> str:
        """Send a 'response' request to the NLU server with per-intent results and return the output text."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port), timeout=5.0
        )
        try:
            context: dict[str, Any] = {
                "session_id": session_id,
                "intents": intents,
            }

            request = json.dumps(
                {"type": "response", "data": {"context": context}},
                separators=(",", ":"),
            )
            payload = f"{request}\n"
            writer.write(payload.encode("utf-8"))
            await writer.drain()

            if writer.can_write_eof():
                writer.write_eof()

            data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            _LOGGER.debug(
                "Sophia NLU response raw (%d bytes): %s", len(data), data[:512]
            )
            text_resp = data.decode("utf-8").strip()
            if not text_resp:
                _LOGGER.error(
                    "Sophia NLU returned empty response to 'response' request"
                )
                return ""
            result = json.loads(text_resp)
            return result.get("data", {}).get("text", "")
        finally:
            writer.close()
            await writer.wait_closed()

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

    async def _build_custom_success(
        self, intent_name: str, slots: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Return a custom success list for intents that need direct state lookup or custom execution.

        Returns None if this intent does not require custom handling.
        """
        # ------------------------------------------------------------------ #
        # HassGetWeather                                                       #
        # ------------------------------------------------------------------ #
        if intent_name == "HassGetWeather":
            name_slot = slots.get("name", {}).get("value", "").strip()
            if name_slot:
                # Look for a weather entity whose friendly name matches
                for state_obj in self.hass.states.async_all("weather"):
                    if state_obj.name.lower() == name_slot.lower():
                        return [self._build_state_entry(state_obj)]
                _LOGGER.warning(
                    "HassGetWeather: no weather entity found with name '%s'", name_slot
                )
                return []
            # Fall back to the default forecast entity
            state_obj = self.hass.states.get("weather.forecast_home")
            if state_obj is not None:
                return [self._build_state_entry(state_obj)]
            _LOGGER.warning("HassGetWeather: weather.forecast_home not found")
            return []

        # ------------------------------------------------------------------ #
        # HassClimateGetTemperature                                            #
        # ------------------------------------------------------------------ #
        if intent_name == "HassClimateGetTemperature":
            name_slot = slots.get("name", {}).get("value", "").strip()
            area_slot = slots.get("area", {}).get("value", "").strip()

            if name_slot:
                # Single climate entity by friendly name
                for state_obj in self.hass.states.async_all("climate"):
                    if state_obj.name.lower() == name_slot.lower():
                        return [self._build_state_entry(state_obj)]
                _LOGGER.warning(
                    "HassClimateGetTemperature: no climate entity found with name '%s'",
                    name_slot,
                )
                return []

            if area_slot:
                # All climate entities in the named area
                area_reg = ar.async_get(self.hass)
                area_entry = area_reg.async_get_area_by_name(area_slot)
                if area_entry is None:
                    _LOGGER.warning(
                        "HassClimateGetTemperature: area '%s' not found", area_slot
                    )
                    return []
                entity_reg = er.async_get(self.hass)
                entity_entries = er.async_entries_for_area(entity_reg, area_entry.id)
                results: list[dict[str, Any]] = []
                for ent in entity_entries:
                    if ent.domain == "climate":
                        state_obj = self.hass.states.get(ent.entity_id)
                        if state_obj is not None:
                            results.append(self._build_state_entry(state_obj))
                return results

            # No slots -- return all climate entities
            return [
                self._build_state_entry(s)
                for s in self.hass.states.async_all("climate")
            ]

        # ------------------------------------------------------------------ #
        # HassTurnOn with domain=automation                                  #
        # ------------------------------------------------------------------ #
        if (
            intent_name == "HassTurnOn"
            and slots.get("domain", {}).get("value") == "automation"
        ):
            name_slot = slots.get("name", {}).get("value", "").strip()
            if name_slot:
                # Find the automation entity whose friendly name matches the slot value
                for state_obj in self.hass.states.async_all("automation"):
                    if state_obj.name.lower() == name_slot.lower():
                        automation_id = state_obj.entity_id
                        try:
                            # Explicitly call the trigger service to execute actions immediately
                            await self.hass.services.async_call(
                                "automation",
                                "trigger",
                                {"entity_id": automation_id},
                                blocking=True,
                            )
                            _LOGGER.debug("Triggered automation '%s' via custom handler", automation_id)
                            return [self._build_state_entry(state_obj)]
                        except Exception as err:
                            _LOGGER.error(
                                "Failed to trigger automation '%s': %s", automation_id, err
                            )
                            return []

                _LOGGER.warning(
                    "HassTurnOn(automation): no automation found with name '%s'", name_slot
                )
                return []
            
            _LOGGER.warning("HassTurnOn(automation): missing 'name' slot to identify automation")
            return []

        # ------------------------------------------------------------------ #
        # HassGetState with domain=person                                      #
        # ------------------------------------------------------------------ #
        if (
            intent_name == "HassGetState"
            and slots.get("domain", {}).get("value") == "person"
        ):
            name_slot = slots.get("name", {}).get("value", "").strip()
            if name_slot:
                for state_obj in self.hass.states.async_all("person"):
                    if state_obj.name.lower() == name_slot.lower():
                        return [self._build_state_entry(state_obj)]
                _LOGGER.warning(
                    "HassGetState(person): no person entity found with name '%s'",
                    name_slot,
                )
                return []
            # No name -- return all person entities
            return [
                self._build_state_entry(s) for s in self.hass.states.async_all("person")
            ]

        return None  # not a custom-handled intent

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

        try:
            results = await self._async_send_to_nlu(text, conversation_id, language)
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

        # HassRespond / HassClarification - immediately provide speech response
        if len(results) == 1:
            only = results[0]
            intent_name = only.get("data", {}).get("intent", {}).get("name")
            
            if intent_name in ("HassRespond", "HassClarification"):
                respond_text = only.get("data", {}).get("text", "").strip()
                respond_slots = {}
                for entity in only.get("data", {}).get("entities", []):
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
                    continue_conversation=(intent_name == "HassClarification") 
                )


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


            # Check whether this intent needs custom success population
            custom_success = await self._build_custom_success(intent_name, slots)
            is_custom = custom_success is not None

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
            if intent_result is None and not is_custom:
                intent_entry: dict[str, Any] = {
                    "response_type": "error",
                    "success": [],
                    "failed": [],
                    "error": intent_err_msg or "unknown error",
                }
                intents_context.append(intent_entry)
                continue

            # Determine success list: custom intents always override HA results
            if is_custom:
                success_list: list[dict[str, Any]] = custom_success  # type: ignore[assignment]
            else:
                # Build success list, enriching entity targets with live HA state and attributes
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
                failed_list = [
                    dataclasses.asdict(t) for t in intent_result.failed_results
                ]
                speech_slots_val = intent_result.speech_slots or None
            else:
                # Custom intent but HA errored -- treat as action_done, no failed targets
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
            if intent_err_msg and not is_custom:
                intent_entry["error"] = intent_err_msg

            intents_context.append(intent_entry)
            if intent_result is not None:
                last_intent_result = intent_result

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
            return conversation.ConversationResult(
                response=last_intent_result, conversation_id=conversation_id
            )

        response.async_set_speech(final_speech or "Sorry, I didn't understand that.")
        return conversation.ConversationResult(
            response=response, conversation_id=conversation_id
        )
