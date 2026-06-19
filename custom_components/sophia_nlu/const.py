"""Constants for the Sophia NLU integration."""

from homeassistant.const import (
    SERVICE_TURN_ON,
    SERVICE_TURN_OFF,
    SERVICE_TOGGLE,
)

DOMAIN = "sophia_nlu"
CONF_HOST = "host"
CONF_PORT = "port"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10520


# Maps intent name -> (domain, service)
INTENT_SERVICE_MAP: dict[str, tuple[str | None, str | None]] = {
    # --- homeassistant core domain routing ---
    "HassTurnOn":                  ("homeassistant", "turn_on"),
    "HassTurnOff":                 ("homeassistant", "turn_off"),
    "HassToggle":                  ("homeassistant", "toggle"),
    "HassSetPosition":             ("cover", "set_cover_position"),  # or valve.set_valve_position via logic
    "HassStopMoving":              ("cover", "stop_cover"),          # or valve.stop_valve via logic

    # --- light ---
    "HassLightSet":                ("light", "turn_on"),

    # --- climate ---
    "HassClimateSetTemperature":   ("climate", "set_temperature"),

    # --- cover ---
    "HassOpenCover":               ("cover", "open_cover"),
    "HassCloseCover":              ("cover", "close_cover"),

    # --- vacuum ---
    "HassVacuumStart":             ("vacuum", "start"),
    "HassVacuumReturnToBase":      ("vacuum", "return_to_base"),
    "HassVacuumCleanArea":         ("vacuum", "start"),

    # --- media_player ---
    "HassMediaPause":              ("media_player", "media_pause"),
    "HassMediaUnpause":            ("media_player", "media_play"),
    "HassMediaNext":               ("media_player", "media_next_track"),
    "HassMediaPrevious":           ("media_player", "media_previous_track"),
    "HassSetVolume":               ("media_player", "volume_set"),
    "HassSetVolumeRelative":       ("media_player", "volume_set"),
    "HassMediaPlayerMute":         ("media_player", "volume_mute"),
    "HassMediaPlayerUnmute":       ("media_player", "volume_mute"),
    "HassMediaSearchAndPlay":      ("media_player", "play_media"),

    # --- fan ---
    "HassFanSetSpeed":             ("fan", "set_percentage"),

    # --- lawn_mower ---
    "HassLawnMowerStartMowing":    ("lawn_mower", "start_mowing"),
    "HassLawnMowerDock":           ("lawn_mower", "dock"),

    # --- humidifier ---
    "HassHumidifierSetpoint":      ("humidifier", "set_humidity"),
    "HassHumidifierMode":          ("humidifier", "set_mode"),
}

_TURN_ON_OVERRIDES: dict[str, tuple[str, str]] = {
    "button":   ("button", "press"),
    "cover":    ("cover", "open_cover"),
    "lock":     ("lock", "lock"),
    "valve":    ("valve", "open_valve"),
    "siren":    ("siren", "turn_on"),
}

_TURN_OFF_OVERRIDES: dict[str, tuple[str, str]] = {
    "cover":    ("cover", "close_cover"),
    "lock":     ("lock", "unlock"),
    "valve":    ("valve", "close_valve"),
}


def get_service_call(intent_name: str, entity_domain: str | None = None) -> tuple[str, str] | None:
    """Returns (domain, service) for an intent based on target entity classification."""
    if intent_name == "HassTurnOn" and entity_domain:
        return _TURN_ON_OVERRIDES.get(entity_domain, ("homeassistant", "turn_on"))

    if intent_name == "HassTurnOff" and entity_domain:
        return _TURN_OFF_OVERRIDES.get(entity_domain, ("homeassistant", "turn_off"))

    # Extra cross-domain handling for specific multi-domain intents
    if intent_name == "HassSetPosition" and entity_domain == "valve":
        return ("valve", "set_valve_position")
    if intent_name == "HassStopMoving" and entity_domain == "valve":
        return ("valve", "stop_valve")

    result = INTENT_SERVICE_MAP.get(intent_name)
    if result is None or result[0] is None or result[1] is None:
        return None
    return (result[0], result[1])





