from __future__ import annotations

from typing import Any, Literal, TypeAlias

import sounddevice as sd

AudioDeviceDirection = Literal["input", "output"]
AudioDeviceOption: TypeAlias = dict[str, str]


def _hostapi_names() -> list[str]:
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        return []
    return [
        str(api.get("name", f"Host API {index}")) for index, api in enumerate(hostapis)
    ]


def _default_device_index(direction: AudioDeviceDirection) -> int | None:
    try:
        input_default, output_default = sd.default.device
    except Exception:
        return None

    raw = input_default if direction == "input" else output_default
    if raw is None:
        return None

    try:
        index = int(raw)
    except (TypeError, ValueError):
        return None

    return None if index < 0 else index


def _format_device_label(
    index: int,
    device: dict[str, Any],
    hostapis: list[str],
) -> str:
    hostapi_index = int(device.get("hostapi", -1) or -1)
    hostapi_name = (
        hostapis[hostapi_index]
        if 0 <= hostapi_index < len(hostapis)
        else "Unknown Host API"
    )
    name = str(device.get("name", f"Device {index}"))
    default_sample_rate = int(float(device.get("default_samplerate", 0) or 0))
    return f"{name} [{hostapi_name}] (#{index}, {default_sample_rate} Hz)"


def list_audio_devices(
    direction: AudioDeviceDirection,
    *,
    include_default: bool = True,
) -> list[AudioDeviceOption]:
    try:
        devices = sd.query_devices()
    except Exception:
        return []

    max_channels_key = (
        "max_input_channels" if direction == "input" else "max_output_channels"
    )
    hostapis = _hostapi_names()

    options: list[AudioDeviceOption] = []

    if include_default:
        default_index = _default_device_index(direction)
        default_label = "System Default"
        if default_index is not None and default_index < len(devices):
            default_label = (
                "System Default: "
                f"{_format_device_label(default_index, devices[default_index], hostapis)}"
            )
        options.append({"value": "", "label": default_label})

    for index, device in enumerate(devices):
        max_channels = int(device.get(max_channels_key, 0) or 0)
        if max_channels <= 0:
            continue
        options.append(
            {
                "value": str(index),
                "label": _format_device_label(index, device, hostapis),
            }
        )

    return options


def list_audio_input_devices(
    *, include_default: bool = True
) -> list[AudioDeviceOption]:
    return list_audio_devices("input", include_default=include_default)


def list_audio_output_devices(
    *,
    include_default: bool = True,
) -> list[AudioDeviceOption]:
    return list_audio_devices("output", include_default=include_default)


def coerce_sounddevice_device(value: str | None) -> int | str | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    if stripped.lstrip("-").isdigit():
        return int(stripped)

    return stripped
