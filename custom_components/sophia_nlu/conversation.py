"""The conversation platform for the Sophia NLU integration."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import ulid

from .const import CONF_HOST, CONF_PORT, DOMAIN

_LOGGER = logging.getLogger(__name__)


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

        # Process each intent returned by the NLU engine.
        # The server may return multiple JSONL lines, one per detected intent.
        speech_parts: list[str] = []
        last_intent_result: intent.IntentResponse | None = None

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
                speech_parts.append(f"NLU error: {error_text}")
                continue

            # Wyoming intent response: {"type": "intent", "data": {"intent": {"name": "...", "confidence": ...}, "entities": [{"name": "...", "value": "..."}], "text": "..."}}
            data = result.get("data", {})
            intent_info = data.get("intent", {})
            intent_name = intent_info.get("name", "")

            if not intent_name:
                _LOGGER.warning(
                    "Sophia NLU returned a result with no intent name, skipping"
                )
                continue

            # Map Wyoming entities to HA intent slots: [{"name": "x", "value": "y"}] -> {"x": {"value": "y"}}
            slots = {}
            for entity in data.get("entities", []):
                slot_name = entity.get("name", "")
                slot_value = entity.get("value", "")
                if slot_name:
                    slots[slot_name] = {"value": slot_value}

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
                )
            except intent.IntentHandleError as err:
                _LOGGER.error("Intent handling error for %s: %s", intent_name, err)
                speech_parts.append(f"Error handling intent: {err}")
                continue
            except intent.IntentUnexpectedError as err:
                _LOGGER.error("Unexpected intent error for %s: %s", intent_name, err)
                speech_parts.append(f"Unexpected error: {err}")
                continue
            except Exception as err:
                _LOGGER.exception("Failed to handle intent %s", intent_name)
                speech_parts.append(f"Failed to handle intent: {err}")
                continue

            last_intent_result = intent_result
            # Collect speech from each successful intent result
            speech = (
                intent_result.speech.get("plain", {}).get("speech", "")
                if intent_result.speech
                else ""
            )
            if speech:
                speech_parts.append(speech)

        # If at least one intent was handled successfully, build a combined response.
        if last_intent_result is not None:
            if speech_parts:
                last_intent_result.async_set_speech(" ".join(speech_parts))
            return conversation.ConversationResult(
                response=last_intent_result, conversation_id=conversation_id
            )

        # All results were errors or had no intent name
        if speech_parts:
            response.async_set_speech(" ".join(speech_parts))
        else:
            response.async_set_speech("Sorry, I didn't understand that.")
        return conversation.ConversationResult(
            response=response, conversation_id=conversation_id
        )
