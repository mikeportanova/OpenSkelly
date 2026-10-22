#!/usr/bin/env python3
"""macOS controller for the 12ft Skelly: BLE over CoreBluetooth, Live audio over A2DP."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

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
CONTROL_SOCKET = Path(__file__).resolve().parent / ".skelly-live" / "mac.sock"
RATE = 44100
TAIL_SECONDS = 0.7  # silence appended to every clip
TTS_MODELS = {
    "kokoro": "mlx-community/Kokoro-82M-bf16",
    "qwen": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
    "clone": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
}
VOICE_REF = CONFIG_FILE.parent / "voice_ref.wav"
REF_TEXT = "Well, well, well. Another little visitor. Come closer, child. I do not bite. I have no lips."
DEFAULTS = {
    # say | kokoro | qwen (VoiceDesign rolls a new voice per line) | clone (every line cloned from `voice design`'s clip)
    "tts": "clone",
    "voice": "Daniel",  # `say -v` voice, or a Kokoro voice such as am_michael
    "voice_instruct": "A very deep, slow, rumbling male voice with a dry, gravelly rasp and a sinister chuckle; an "
                      "ancient giant skeleton, theatrical and menacing like a horror movie villain, drawn-out words, "
                      "cavernous and hollow.",
    "voice_seed": 42,  # VoiceDesign seed used by `voice design` to render the reference clip
    "voice_ref_text": REF_TEXT,
    # ffmpeg chain before the jaw: slight pitch-down, cavern echo, then level for the mouth threshold.
    # Every filter here is stream-safe (no loudnorm: it needs the whole clip and the voice is streamed).
    "voice_fx": "asetrate=44100*0.9,aresample=44100,aecho=0.8:0.85:60|130:0.45|0.25,"
                "acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=8,volume=5dB,alimiter=limit=0.9:attack=5:release=60",
    "model": "claude-haiku-5-5",  # chat model; Haiku answers fastest, which is what a kid at the door needs
    "filler": True,  # play a short cloned interjection when the reply takes longer than FILLER_AFTER
    "persona": None,  # what Skelly is like and how he answers; None = talk.PERSONA. `instruct` only shapes the voice sound
    "whisper_model": "mlx-community/whisper-small-mlx",
    "vad_threshold": 0.02,  # hands-free: RMS above this is speech (talk --listen-test shows the room)
    "vad_silence": 1.2,  # hands-free: seconds of quiet that end a kid's turn
    "mic_device": None,  # input device name or substring, e.g. "Yeti"; None = system default
}


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


def bt_paired(mac: str) -> bool:
    return re.search(r"(?<!not )paired", run("blueutil", "--info", mac, timeout=10)) is not None


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


def output_device_index(name: str = LIVE_NAME, wait: float = 20) -> int:
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


def default_outputs() -> dict[str, str]:
    """Current macOS default output and alert devices, by SwitchAudioSource type."""
    return {kind: run("SwitchAudioSource", "-c", "-t", kind).strip() for kind in ("output", "system")}


def restore_default_outputs(previous: dict[str, str]) -> None:
    """Keep every other Mac sound on the speakers: only our explicit playback targets the Skelly."""
    for kind, name in previous.items():
        if name and LIVE_NAME not in name:
            run("SwitchAudioSource", "-s", name, "-t", kind)


async def connect_live(skelly: Skelly, live_mac: str) -> int:
    """Run the DecorPro Live handshake; returns the Skelly output device index."""
    previous = default_outputs()  # macOS will switch these to the Skelly when it connects
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
    run("SwitchAudioSource", "-s", LIVE_NAME, "-t", "output")
    run("osascript", "-e", "set volume output volume 100")  # acts on the default device: a fresh BT sink sits at 50
    restore_default_outputs(previous)
    await skelly.write(COMMANDS["start_live"])
    await asyncio.sleep(0.08)
    await skelly.write(COMMANDS["volume_100"])
    return device


async def disconnect_live(skelly: Skelly, live_mac: str) -> None:
    try:
        await skelly.write(COMMANDS["stop_live"])
    finally:
        run("blueutil", "--disconnect", live_mac, timeout=10)


def decode_audio(source: Path | bytes, filters: str | None, in_rate: int | None = None, pad: bool = True) -> np.ndarray:
    """ffmpeg -> 44.1 kHz stereo int16; `source` is a file, or raw mono float32 PCM at `in_rate`."""
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if isinstance(source, bytes):
        args += ["-f", "f32le", "-ar", str(in_rate), "-ac", "1", "-i", "-"]
    else:
        args += ["-i", str(source)]
    # resample first so a chain written for 44.1 kHz (asetrate=44100*0.9) means the same for 24 kHz TTS output;
    # pad the end so the Bluetooth buffer flushes the last word and the echo tail instead of clipping it
    args += ["-af", f"aresample={RATE}" + (f",{filters}" if filters else "") + (f",apad=pad_dur={TAIL_SECONDS}" if pad else "")]
    args += ["-f", "s16le", "-ar", str(RATE), "-ac", "2", "-"]
    raw = subprocess.run(args, check=True, capture_output=True, timeout=120,
                         input=source if isinstance(source, bytes) else None).stdout
    return np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)


def play_pcm(pcm: np.ndarray, device: int) -> None:
    sd.play(pcm, RATE, device=device, blocking=True)


_tts_model: object = None
_tts_executor = ThreadPoolExecutor(max_workers=1)  # MLX generation is not reentrant: one sentence at a time
STREAM_RATE = 24000  # mlx TTS output
FILLER_AFTER = 1.2  # seconds of silence after the kid stops before a filler covers the wait
FILLER_DIR = CONFIG_FILE.parent / "fillers"
FILLERS = ["Hmmmm.", "Well now.", "Ooooh.", "My, my.", "Ahh.", "Oh ho.", "Hmm, yes.", "Ha!", "Let me see.", "Mmm."]
MODELS = {"Haiku (fastest)": "claude-haiku-5-5", "Sonnet": "claude-sonnet-5-5", "Opus (wittiest)": "claude-opus-5-5"}


def synthesize(text: str, config: dict) -> tuple[Path | bytes, int | None]:
    """Text -> audio with the configured engine: a wav path, or (mono float32 PCM, rate); blocking."""
    global _tts_model
    if config["tts"] == "say":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        subprocess.run(["say", "-v", config["voice"], "-o", str(path), f"--data-format=LEI16@{RATE}", "--", text],
                       check=True, timeout=60)
        return path, None
    from mlx_audio.tts.utils import load_model

    if _tts_model is None:
        _tts_model = load_model(TTS_MODELS[config["tts"]])
    if config["tts"] == "kokoro":
        results = _tts_model.generate(text, voice=config["voice"], speed=0.85, lang_code="a")
    elif config["tts"] == "clone":
        if not VOICE_REF.exists():
            raise RuntimeError(f"no reference voice at {VOICE_REF}; run: skelly_mac.py voice design")
        results = _tts_model.generate(text, ref_audio=str(VOICE_REF), ref_text=config["voice_ref_text"], lang_code="en")
    else:
        import mlx.core as mx

        mx.random.seed(int(config["voice_seed"]))
        results = _tts_model.generate(text, instruct=config["voice_instruct"], lang_code="en")
    chunks, rate = [], 24000
    for result in results:
        chunks.append(np.array(result.audio, dtype=np.float32))
        rate = result.sample_rate
    return np.concatenate(chunks).tobytes(), rate


def synthesize_stream(text: str, config: dict) -> Iterator[bytes]:
    """Clone/design TTS as mono float32 chunks at STREAM_RATE, yielded while the model is still generating."""
    global _tts_model
    from mlx_audio.tts.utils import load_model

    if _tts_model is None:
        _tts_model = load_model(TTS_MODELS[config["tts"]])
    if config["tts"] == "clone":
        results = _tts_model.generate(text, ref_audio=str(VOICE_REF), ref_text=config["voice_ref_text"], lang_code="en",
                                      stream=True, streaming_interval=1.0)
    else:
        import mlx.core as mx

        mx.random.seed(int(config["voice_seed"]))
        results = _tts_model.generate(text, instruct=config["voice_instruct"], lang_code="en", stream=True, streaming_interval=1.0)
    for result in results:
        yield np.array(result.audio, dtype=np.float32).tobytes()


def fx_args(filters: str | None, in_rate: int) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(in_rate), "-ac", "1", "-i", "-",
            "-af", f"aresample={RATE}" + (f",{filters}" if filters else "") + f",apad=pad_dur={TAIL_SECONDS}",
            "-f", "s16le", "-ar", str(RATE), "-ac", "2", "-"]


class StreamJob:
    """One sentence: TTS chunks -> ffmpeg fx (live) -> the jaw, playing while the rest is still generating."""

    def __init__(self, text: str, config: dict) -> None:
        self.text = text
        self.config = config
        self.proc = subprocess.Popen(fx_args(config["voice_fx"] or None, STREAM_RATE), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.started = asyncio.get_running_loop().create_future()  # resolved when the first chunk exists
        self.generation = asyncio.get_running_loop().run_in_executor(_tts_executor, self._generate)

    def _generate(self) -> None:
        loop = self.started.get_loop()
        try:
            for chunk in synthesize_stream(self.text, self.config):
                self.proc.stdin.write(chunk)
                if not self.started.done():
                    loop.call_soon_threadsafe(lambda: self.started.done() or self.started.set_result(True))
        except Exception as exc:
            loop.call_soon_threadsafe(lambda: self.started.done() or self.started.set_exception(exc))
            raise
        finally:
            self.proc.stdin.close()

    def frames(self, frame_bytes: int = RATE * 4 // 10) -> Iterator[bytes]:
        """Processed stereo int16 PCM in ~100 ms pieces, frame-aligned."""
        rest = b""
        while chunk := self.proc.stdout.read(frame_bytes):
            data = rest + chunk
            cut = len(data) // 4 * 4
            yield data[:cut]
            rest = data[cut:]
        self.proc.wait()

    def play(self, device: int) -> None:
        with sd.RawOutputStream(samplerate=RATE, channels=2, dtype="int16", device=device) as out:
            for frame in self.frames():
                out.write(frame)


def filler_pcm(config: dict, text: str) -> np.ndarray:
    """A rendered filler line, cached on disk per voice reference and fx chain."""
    stamp = f"{text}|{VOICE_REF.stat().st_mtime if VOICE_REF.exists() else 0}|{config['tts']}|{config['voice_fx']}"
    path = FILLER_DIR / (hashlib.sha1(stamp.encode()).hexdigest() + ".npy")
    if path.exists():
        return np.load(path)
    source, rate = synthesize(text, config)
    pcm = decode_audio(source, config["voice_fx"] or None, rate, pad=False)  # the real answer follows at once
    FILLER_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, pcm)
    return pcm


async def render(text: str, config: dict) -> np.ndarray:
    """TTS -> ffmpeg effects -> PCM ready for the jaw."""
    source, rate = await asyncio.to_thread(synthesize, text, config)
    try:
        return await asyncio.to_thread(decode_audio, source, config["voice_fx"] or None, rate)
    finally:
        if isinstance(source, Path):
            source.unlink(missing_ok=True)


async def speak(text: str, config: dict, device: int) -> None:
    await asyncio.to_thread(play_pcm, await render(text, config), device)


# ---------------------------------------------------------------- resident server


class Session:
    """One long-lived Live connection shared by the talk loop and socket commands."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.skelly: Skelly | None = None
        self.device = -1
        self.lock = asyncio.Lock()  # whoever holds it owns the jaw
        self.listener: object = None  # set by talk.run; muted while socket commands speak

    async def ensure_connected(self) -> None:
        if self.skelly and self.skelly.client.is_connected and bt_connected(self.config["live_mac"]):
            return
        if self.skelly:
            await self.skelly.__aexit__()
        print("Connecting to Skelly…", flush=True)
        self.skelly = await Skelly(self.config["ble_address"]).__aenter__()
        self.device = await connect_live(self.skelly, self.config["live_mac"])
        await asyncio.sleep(1.5)  # let the connection chime finish
        print("Skelly Live is up", flush=True)

    async def close(self) -> None:
        if self.skelly:
            try:
                await disconnect_live(self.skelly, self.config["live_mac"])
            finally:
                await self.skelly.__aexit__()
                self.skelly = None

    async def handle(self, request: dict) -> None:
        async with self.lock:
            await self.ensure_connected()
            if self.listener:
                self.listener.mute()
            try:
                if request["cmd"] == "say":
                    await speak(request["text"], self.config, self.device)
                elif request["cmd"] == "play":
                    pcm = decode_audio(Path(request["path"]), "loudnorm=I=-10:TP=-1:LRA=11")
                    await asyncio.to_thread(play_pcm, pcm, self.device)
                elif request["cmd"] == "frames":
                    for frame in request["frames"]:
                        await self.skelly.write(frame)
                else:
                    raise ValueError(f"unknown command {request['cmd']!r}")
            finally:
                if self.listener:
                    await asyncio.sleep(0.3)
                    self.listener.unmute()


async def serve(config: dict, talk_mode: bool, vad: bool = False) -> int:
    session = Session(config)
    stop = asyncio.Event()

    async def on_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = json.loads(await reader.readline())
            if request.get("cmd") == "stop":
                stop.set()
            else:
                await session.handle(request)
            response = {"ok": True}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()
        writer.close()

    async def keepalive() -> None:
        while True:
            await asyncio.sleep(3)
            async with session.lock:
                try:
                    await session.ensure_connected()
                except Exception as exc:
                    print(f"reconnect failed: {exc}", flush=True)

    CONTROL_SOCKET.parent.mkdir(exist_ok=True)
    CONTROL_SOCKET.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(on_client, path=str(CONTROL_SOCKET))
    tasks = [asyncio.create_task(keepalive())]
    try:
        await session.ensure_connected()
        if talk_mode:
            import talk

            talking = asyncio.create_task(talk.run(session, vad))
            tasks.append(talking)
            talking.add_done_callback(lambda _t: stop.set())
        else:
            await asyncio.to_thread(synthesize, "Boo.", config)  # load the voice model before the first `say`
            print("Skelly server is running; say/play/lights/eyes now go through it. Ctrl-C or `stop` to end.", flush=True)
        await stop.wait()
    finally:
        for task in tasks:
            task.cancel()
        server.close()
        CONTROL_SOCKET.unlink(missing_ok=True)
        await session.close()
    return 0


def send_to_server(request: dict) -> bool:
    """Forward a command to a running server; False when none is listening."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(str(CONTROL_SOCKET))
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    with client:
        client.sendall(json.dumps(request).encode() + b"\n")
        response = json.loads(client.makefile().readline())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "server command failed"))
    return True


# ---------------------------------------------------------------- one-shot commands (fallback when no server runs)


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
            print(f"Found {LIVE_NAME} at {live_mac}; pairing…", flush=True)
            pair = ["blueutil", "--pair", live_mac] + ([args.pin] if args.pin else [])
            try:  # stdin closed so a PIN prompt fails fast instead of hanging invisibly
                subprocess.run(pair, stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                pass
            if not bt_paired(live_mac):
                print(f"Automatic pairing failed. Open System Settings > Bluetooth, pair {LIVE_NAME} "
                      "(enter the PIN from the box or the DecorPro app if asked), then press Enter here.")
                await asyncio.to_thread(input)
                if not bt_paired(live_mac):
                    print(f"{LIVE_NAME} is still not paired; retry with: configure --pin <PIN>", file=sys.stderr)
                    return 1
            run("blueutil", "--disconnect", live_mac)
    config = load_config() if CONFIG_FILE.exists() else dict(DEFAULTS)
    config.update({"ble_address": address, "live_mac": live_mac, "voice": args.voice or config["voice"]})
    if args.voice_fx is not None:
        config["voice_fx"] = args.voice_fx
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {CONFIG_FILE}\n{json.dumps(config, indent=2)}")
    return 0


def cmd_voice_design(args: argparse.Namespace) -> int:
    """Render the reference clip with VoiceDesign (seeded) that `clone` mode copies for every line."""
    config = load_config()
    if args.seed is not None:
        config["voice_seed"] = args.seed
    if args.instruct:
        config["voice_instruct"] = args.instruct
    design = {**config, "tts": "qwen"}
    print(f"Rendering reference voice (seed {design['voice_seed']})…", flush=True)
    pcm, rate = synthesize(config["voice_ref_text"], design)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1",
                    "-i", "-", str(VOICE_REF)], input=pcm, check=True)
    config.update({"tts": "clone", "voice_seed": design["voice_seed"], "voice_instruct": design["voice_instruct"]})
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {VOICE_REF}; tts=clone. Restart serve/talk to use it.")
    return 0


async def cmd_frames(frames: list[str]) -> int:
    if send_to_server({"cmd": "frames", "frames": frames}):
        return 0
    async with Skelly(load_config()["ble_address"]) as skelly:
        for frame in frames:
            await skelly.write(frame)
    return 0


async def cmd_play(path: Path) -> int:
    if send_to_server({"cmd": "play", "path": str(path.resolve())}):
        return 0
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
    if send_to_server({"cmd": "say", "text": text}):
        return 0
    config = load_config()
    async with Skelly(config["ble_address"]) as skelly:
        device = await connect_live(skelly, config["live_mac"])
        try:
            await asyncio.sleep(1.5)  # let the connection chime finish before the first word
            await speak(text, config, device)
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
    return await serve(load_config(), talk_mode=True, vad=args.vad)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    setup = subs.add_parser("configure", help="find Skelly, pair its audio endpoint, save mac.json")
    setup.add_argument("--ble-address")
    setup.add_argument("--live-mac")
    setup.add_argument("--pin")
    setup.add_argument("--voice", help="macOS `say` voice (see: say -v '?')")
    setup.add_argument("--voice-fx", help="ffmpeg audio filter chain applied to the voice; empty string for none")
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
    voice = subs.add_parser("voice", help="manage the cloned skeleton voice")
    voice_subs = voice.add_subparsers(dest="voice_action", required=True)
    design = voice_subs.add_parser("design", help="render the reference clip that clone mode copies")
    design.add_argument("--seed", type=int, help="VoiceDesign seed; different seeds give different skeletons")
    design.add_argument("--instruct", help="voice description for VoiceDesign")
    say.add_argument("text")
    subs.add_parser("serve", help="hold the Live connection; say/play/lights/eyes go through it until `stop`")
    subs.add_parser("stop", help="stop a running serve/talk and disconnect")
    talk = subs.add_parser("talk", help="serve + two-way conversation: mic -> Whisper -> Claude -> Skelly")
    talk.add_argument("--vad", action="store_true", help="hands-free voice detection instead of the push-to-talk window")
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
        if args.action == "voice":
            return cmd_voice_design(args)
        if args.action == "talk":
            return asyncio.run(cmd_talk(args))
        if args.action == "serve":
            return asyncio.run(serve(load_config(), talk_mode=False))
        if args.action == "stop":
            print("stopped" if send_to_server({"cmd": "stop"}) else "no server running")
            return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"skelly_mac failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
