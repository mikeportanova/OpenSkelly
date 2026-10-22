"""Skelly BLE frame builders shared by the Linux and macOS controllers."""

from __future__ import annotations

import argparse

COMMANDS = {
    "stop_file": "aac60000000000000000ed",
    "enable_live_radio": "aafd0100000000000000d1",
    "select_bluetooth": "aafe010000000000000024",
    "start_live": "aafc010000000000000075",
    "stop_live": "aafc000000000000000036",
    "volume_100": "aafa640000000000000031",
}
ACK_LIVE_RADIO = "bbfd01"
ACK_BLUETOOTH = "bbfe01"


def crc8(data: bytes) -> int:
    """DecorPro's reflected CRC-8 (polynomial 0x8C)."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
    return crc


def protocol_frame(opcode: int, payload: bytes) -> str:
    frame = bytes((0xAA, opcode)) + payload
    return (frame + bytes((crc8(frame),))).hex()


def light_color_frame(channel: int, rgb: str) -> str:
    return protocol_frame(0xF4, bytes((channel,)) + bytes.fromhex(rgb) + b"\x00\xff\xff\xff\xff\x00")


def light_brightness_frame(channel: int, brightness: int) -> str:
    return protocol_frame(0xF3, bytes((channel, brightness)) + b"\xff\xff\xff\xff\x00\x00")


def eye_frame(pattern: int) -> str:
    return protocol_frame(0xF9, bytes((pattern, 0)) + b"\xff\xff\xff\xff\x00")


def light_channel(value: str) -> int:
    aliases = {"chest": 0, "all": 0xFF}
    if value.lower() in aliases:
        return aliases[value.lower()]
    try:
        channel = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channel must be chest, all, or 0-5") from exc
    if not 0 <= channel <= 5:
        raise argparse.ArgumentTypeError("channel must be chest, all, or 0-5")
    return channel
