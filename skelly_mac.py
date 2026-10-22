#!/usr/bin/env python3
"""macOS controller for the 12ft Skelly: BLE over CoreBluetooth, Live audio over A2DP."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from bleak import BleakClient, BleakScanner

from protocol import (
    ACK_BLUETOOTH,
    ACK_LIVE_RADIO,
    COMMANDS,
    eye_frame,
    light_brightness_frame,
    light_channel,
    light_color_frame,
)

AE01 = "0000ae01-0000-1000-8000-00805f9b34fb"
AE02 = "0000ae02-0000-1000-8000-00805f9b34fb"
BLE_NAME = "12ft Skelly"
LIVE_NAME = "12ft Skelly(Live)"
CONFIG_FILE = Path.home() / ".config" / "openskelly" / "mac.json"
RATE = 44100
DEFAULTS = {"voice": "Daniel", "whisper_model": "mlx-community/whisper-small-mlx", "vad_threshold": 0.02}


def load_config() -> dict:
    try:
        config = json.loads(CONFIG_FILE.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError("not configured; run: skelly_mac.py configure") from exc
    return {**DEFAULTS, **config}


def run(*args: str, timeout: float = 30) -> str:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout).stdout


def normalize_mac(value: str) -> str:
    return value.replace("-", ":").upper()


def paired_live_mac(paired_output: str) -> str | None:
    for line in paired_output.splitlines():
        match = re.match(r"address: ([0-9a-f-]{17}),.*name: \"(.+?)\"", line, re.IGNORECASE)
        if match and match.group(2) == LIVE_NAME:
            return normalize_mac(match.group(1))
    return None


def bt_connected(mac: str) -> bool:
    return run("blueutil", "--is-connected", mac, timeout=10).strip() == "1"


class Skelly:
    """One BLE control session: `async with Skelly(address) as skelly: await skelly.write(...)`."""

    def __init__(self, address: str) -> None:
        self.client = BleakClient(address, timeout=20)
        self.notifications: list[str] = []

    async def __aenter__(self) -> Skelly:
        await self.client.connect()
        await self.client.start_notify(AE02, self._on_notify)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        try:
            await self.client.disconnect()
        except Exception:
            pass

    def _on_notify(self, _char: object, data: bytearray) -> None:
        self.notifications.append(bytes(data).hex())

    async def write(self, frame: str, ack: str | None = None, timeout: float = 5.0) -> None:
        self.notifications.clear()
        await self.client.write_gatt_char(AE01, bytes.fromhex(frame), response=False)
        if ack is None:
            await asyncio.sleep(0.12)
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(n.startswith(ack) for n in self.notifications):
                return
            await asyncio.sleep(0.05)
        raise RuntimeError(f"timed out waiting for Skelly ack {ack}")


async def find_ble_address(timeout: float = 15) -> str:
    device = await BleakScanner.find_device_by_filter(
        lambda _d, ad: (ad.local_name or "") == BLE_NAME, timeout=timeout
    )
    if not device:
        raise RuntimeError(f"no BLE device named {BLE_NAME!r} in range (is Skelly powered on?)")
    return device.address


def output_device_index(name: str = LIVE_NAME, wait: float = 10) -> int:
    deadline = time.monotonic() + wait
    while True:
        sd._terminate()  # ponytail: PortAudio caches the device list; re-init to see a new BT sink
        sd._initialize()
        for index, device in enumerate(sd.query_devices()):
            if name in device["name"] and device["max_output_channels"] > 0:
                return index
        if time.monotonic() >= deadline:
            raise RuntimeError(f"audio output {name!r} did not appear; check System Settings > Bluetooth")
        time.sleep(0.5)


async def connect_live(skelly: Skelly, live_mac: str) -> int:
    """Run the DecorPro Live handshake; returns the Skelly output device index."""
    await skelly.write(COMMANDS["stop_file"])
    await skelly.write(COMMANDS["stop_live"])
    await skelly.write(COMMANDS["enable_live_radio"], ack=ACK_LIVE_RADIO)
    for _ in range(4):
        run("blueutil", "--connect", live_mac, timeout=25)
        if bt_connected(live_mac):
            break
        await asyncio.sleep(1)
        await skelly.write(COMMANDS["enable_live_radio"])
    else:
        raise RuntimeError(f"could not connect {LIVE_NAME}; pair it first with: skelly_mac.py configure")
    await skelly.write(COMMANDS["select_bluetooth"], ack=ACK_BLUETOOTH)
    device = output_device_index()
    await skelly.write(COMMANDS["start_live"])
    await asyncio.sleep(0.08)
    await skelly.write(COMMANDS["volume_100"])
    return device


async def disconnect_live(skelly: Skelly, live_mac: str) -> None:
    try:
        await skelly.write(COMMANDS["stop_live"])
    finally:
        run("blueutil", "--disconnect", live_mac, timeout=10)


def decode_audio(path: Path, filters: str | None) -> np.ndarray:
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if filters:
        args += ["-af", filters]
    args += ["-f", "s16le", "-ar", str(RATE), "-ac", "2", "-"]
    raw = subprocess.run(args, check=True, capture_output=True, timeout=120).stdout
    return np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)


def play_pcm(pcm: np.ndarray, device: int) -> None:
    sd.play(pcm, RATE, device=device, blocking=True)


def say_device(listing: str, name: str = LIVE_NAME) -> str:
    """Pick the `say -a` device id whose name contains `name` from `say -a '?'` output."""
    for line in listing.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)", line)
        if match and name in match.group(2):
            return match.group(1)
    raise RuntimeError(f"`say` cannot see audio output {name!r}")


async def speak(text: str, voice: str) -> None:
    """macOS TTS straight into the Skelly sink; the jaw follows the audio level."""
    device = say_device(run("say", "-a", "?"))
    proc = await asyncio.create_subprocess_exec("say", "-a", device, "-v", voice, "--", text)
    if await proc.wait() != 0:
        raise RuntimeError("say failed; is the Skelly audio device connected?")


# ---------------------------------------------------------------- commands


async def cmd_configure(args: argparse.Namespace) -> int:
    address = args.ble_address or await find_ble_address()
    live_mac = args.live_mac or paired_live_mac(run("blueutil", "--paired"))
    if not live_mac:
        print("Enabling Skelly's classic radio and scanning for it (up to 20 s)…")
        async with Skelly(address) as skelly:
            await skelly.write(COMMANDS["stop_file"])
            await skelly.write(COMMANDS["enable_live_radio"], ack=ACK_LIVE_RADIO)
            live_mac = paired_live_mac(run("blueutil", "--inquiry", "20", timeout=40))
            if not live_mac:
                print(f"{LIVE_NAME} not found; pair it in System Settings > Bluetooth, then rerun.", file=sys.stderr)
                return 1
            pair = ["blueutil", "--pair", live_mac] + ([args.pin] if args.pin else [])
            run(*pair, timeout=60)
            if run("blueutil", "--is-paired", live_mac).strip() != "1":
                print(f"pairing failed; pair {LIVE_NAME} ({live_mac}) in System Settings > Bluetooth.", file=sys.stderr)
                return 1
            run("blueutil", "--disconnect", live_mac)
    config = load_config() if CONFIG_FILE.exists() else dict(DEFAULTS)
    config.update({"ble_address": address, "live_mac": live_mac, "voice": args.voice or config["voice"]})
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {CONFIG_FILE}\n{json.dumps(config, indent=2)}")
    return 0


async def cmd_frames(frames: list[str]) -> int:
    async with Skelly(load_config()["ble_address"]) as skelly:
        for frame in frames:
            await skelly.write(frame)
    return 0


async def cmd_play(path: Path) -> int:
    config = load_config()
    print("Preparing audio…", flush=True)
    pcm = decode_audio(path, "loudnorm=I=-10:TP=-1:LRA=11,adelay=2000|2000,apad=pad_dur=0.75")
    async with Skelly(config["ble_address"]) as skelly:
        device = await connect_live(skelly, config["live_mac"])
        try:
            await asyncio.to_thread(play_pcm, pcm, device)
        finally:
            await disconnect_live(skelly, config["live_mac"])
    return 0


async def cmd_say(text: str) -> int:
    config = load_config()
    async with Skelly(config["ble_address"]) as skelly:
        await connect_live(skelly, config["live_mac"])
        try:
            await asyncio.sleep(1.5)  # let the connection chime finish before the first word
            await speak(text, config["voice"])
        finally:
            await disconnect_live(skelly, config["live_mac"])
    return 0


async def cmd_talk(args: argparse.Namespace) -> int:
    import talk

    config = load_config() if CONFIG_FILE.exists() else dict(DEFAULTS)
    if args.threshold:
        config["vad_threshold"] = args.threshold
    if args.listen_test:
        return talk.listen_test(config)
    config = load_config()
    async with Skelly(config["ble_address"]) as skelly:
        await connect_live(skelly, config["live_mac"])
        try:
            await talk.run(config)
        finally:
            await disconnect_live(skelly, config["live_mac"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    setup = subs.add_parser("configure", help="find Skelly, pair its audio endpoint, save mac.json")
    setup.add_argument("--ble-address")
    setup.add_argument("--live-mac")
    setup.add_argument("--pin")
    setup.add_argument("--voice", help="macOS `say` voice (see: say -v '?')")
    lights = subs.add_parser("lights", help="control Skelly's RGB chest light")
    light_subs = lights.add_subparsers(dest="lights_action", required=True)
    color = light_subs.add_parser("color")
    color.add_argument("rgb", help="six hex digits, e.g. ff0000")
    color.add_argument("--channel", type=light_channel, default=0)
    color.add_argument("--brightness", type=int, choices=range(101), default=100, metavar="0-100")
    brightness = light_subs.add_parser("brightness")
    brightness.add_argument("value", type=int, choices=range(101), metavar="0-100")
    brightness.add_argument("--channel", type=light_channel, default=0)
    off = light_subs.add_parser("off")
    off.add_argument("--channel", type=light_channel, default=0)
    eyes = subs.add_parser("eyes", help="select a built-in eye pattern")
    eyes.add_argument("pattern", type=int, choices=range(1, 21), metavar="1-20")
    play = subs.add_parser("play", help="play an audio file through Skelly with jaw animation")
    play.add_argument("audio", type=Path)
    say = subs.add_parser("say", help="speak a line through Skelly")
    say.add_argument("text")
    talk = subs.add_parser("talk", help="two-way conversation: mic -> Whisper -> Claude -> Skelly")
    talk.add_argument("--threshold", type=float, help="VAD RMS threshold (see --listen-test)")
    talk.add_argument("--listen-test", action="store_true", help="print mic levels for 5 s and exit")
    args = parser.parse_args()
    try:
        if args.action == "configure":
            return asyncio.run(cmd_configure(args))
        if args.action == "lights":
            if args.lights_action == "color":
                if not re.fullmatch(r"[0-9a-fA-F]{6}", args.rgb):
                    parser.error("RGB color must be exactly six hexadecimal digits")
                frames = [light_color_frame(args.channel, args.rgb), light_brightness_frame(args.channel, args.brightness)]
            elif args.lights_action == "brightness":
                frames = [light_brightness_frame(args.channel, args.value)]
            else:
                frames = [light_brightness_frame(args.channel, 0)]
            return asyncio.run(cmd_frames(frames))
        if args.action == "eyes":
            return asyncio.run(cmd_frames([eye_frame(args.pattern)]))
        if args.action == "play":
            return asyncio.run(cmd_play(args.audio))
        if args.action == "say":
            return asyncio.run(cmd_say(args.text))
        if args.action == "talk":
            return asyncio.run(cmd_talk(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"skelly_mac failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
