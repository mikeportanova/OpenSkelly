"""Kids talk to Skelly; Whisper hears them, Claude answers, the jaw moves."""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from typing import AsyncIterator

import numpy as np
import sounddevice as sd

MIC_RATE = 16000  # what Whisper wants; the mic is opened at its own rate and resampled
BLOCK = 480  # 30 ms at MIC_RATE; a Listener scales it to its device rate
MODEL = "claude-opus-5-5"
PERSONA = (
    "You are Skelly, a 12-foot-tall skeleton standing in a front yard on Halloween night, chatting with "
    "trick-or-treaters, mostly young kids. Be playful and a little spooky, never actually frightening, mean, "
    "or gross. Everything you say is spoken aloud through your jaw, so answer in one or two short sentences "
    "a child can follow: no lists, no emoji, no stage directions. Respond to what the kid actually said and "
    "stop there; do not tack a question onto the end of your answer. If a kid tells you their name or "
    "costume, use it. If asked something you should not answer, deflect with a bone pun."
)


class Segmenter:
    """Energy-based VAD: feed 30 ms blocks, get back a finished utterance or None."""

    def __init__(self, threshold: float = 0.02, start_blocks: int = 3, silence_blocks: int = 27,
                 min_blocks: int = 14, max_blocks: int = 500, preroll: int = 10) -> None:
        self.threshold = threshold
        self.start_blocks, self.silence_blocks = start_blocks, silence_blocks
        self.min_blocks, self.max_blocks = min_blocks, max_blocks
        self.preroll: deque[np.ndarray] = deque(maxlen=preroll)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.loud = 0
        self.silent = 0
        self.buffer: list[np.ndarray] = []
        self.preroll.clear()

    def feed(self, block: np.ndarray) -> np.ndarray | None:
        loud = float(np.sqrt(np.mean(block**2))) > self.threshold
        if not self.active:
            self.preroll.append(block)
            self.loud = self.loud + 1 if loud else 0
            if self.loud >= self.start_blocks:
                self.active, self.silent, self.buffer = True, 0, list(self.preroll)
            return None
        self.buffer.append(block)
        self.silent = 0 if loud else self.silent + 1
        if self.silent < self.silence_blocks and len(self.buffer) < self.max_blocks:
            return None
        utterance = np.concatenate(self.buffer) if len(self.buffer) - self.silent >= self.min_blocks else None
        self.reset()
        return utterance


def split_sentences(text: str) -> tuple[list[str], str]:
    """Return the complete sentences in `text` and the unfinished remainder."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


def to_whisper_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    """Resample mono float32 PCM to MIC_RATE (USB mics such as a Blue Yeti only run at 44.1/48 kHz)."""
    if rate == MIC_RATE:
        return audio
    import subprocess

    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1",
                          "-i", "-", "-f", "f32le", "-ar", str(MIC_RATE), "-ac", "1", "-"],
                         input=audio.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


class Listener:
    def __init__(self, threshold: float, device: int | str | None = None) -> None:
        self.segmenter = Segmenter(threshold)
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.muted = False
        self.loop = asyncio.get_running_loop()
        self.rate = int(sd.query_devices(device, "input")["default_samplerate"])
        self.stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="float32",
                                     blocksize=BLOCK * self.rate // MIC_RATE, device=device, callback=self._callback)

        self.recording: list[np.ndarray] | None = None

    def _callback(self, indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        if self.recording is not None:
            self.recording.append(indata[:, 0].copy())
        elif not self.muted:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, indata[:, 0].copy())

    def begin(self) -> None:
        self.recording = []

    def end(self) -> np.ndarray | None:
        """Stop push-to-talk recording; None when it was too short to be a word."""
        blocks, self.recording = self.recording or [], None
        if sum(len(b) for b in blocks) < self.rate * 0.3:
            return None
        return to_whisper_rate(np.concatenate(blocks), self.rate)

    def mute(self) -> None:
        self.muted = True

    def unmute(self) -> None:
        while not self.queue.empty():
            self.queue.get_nowait()
        self.segmenter.reset()
        self.muted = False

    async def next_utterance(self) -> np.ndarray:
        while True:
            utterance = self.segmenter.feed(await self.queue.get())
            if utterance is not None:
                return to_whisper_rate(utterance, self.rate)


def transcribe(audio: np.ndarray, model: str) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="en", fp16=True,
                                    condition_on_previous_text=False, initial_prompt="Talking to Skelly the skeleton.")
    return str(result["text"]).strip()


class Brain:
    def __init__(self, model: str = MODEL, persona: str = PERSONA) -> None:
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic()
        self.model = model
        self.persona = persona
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages.clear()

    async def reply(self, text: str) -> AsyncIterator[str]:
        """Stream the answer back one sentence at a time."""
        self.messages.append({"role": "user", "content": text})
        buffer = ""
        async with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=300,
            system=[{"type": "text", "text": self.persona, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for chunk in stream.text_stream:
                buffer += chunk
                done, buffer = split_sentences(buffer)
                for sentence in done:
                    yield sentence
            final = await stream.get_final_message()
        if buffer.strip():
            yield buffer.strip()
        if final.stop_reason == "refusal" or not any(b.type == "text" for b in final.content):
            self.messages.pop()
            yield "Hmm, that one rattled my skull. Ask me something else!"
            return
        self.messages.append({"role": "assistant", "content": final.content})
        del self.messages[:-40]


async def run(session: object, vad: bool = False) -> None:
    """Conversation loop on a skelly_mac.Session: push-to-talk window by default, energy VAD with vad=True."""
    from skelly_mac import speak, synthesize

    config = session.config
    window = None if vad else __import__("ui").TalkWindow()
    print("Loading speech models (first run downloads them)…", flush=True)
    await asyncio.to_thread(transcribe, np.zeros(MIC_RATE, dtype=np.float32), config["whisper_model"])
    await asyncio.to_thread(synthesize, "Boo.", config)
    brain = Brain(config.get("model", MODEL), config.get("persona") or PERSONA)
    listener = Listener(config["vad_threshold"], config.get("mic_device"))
    session.listener = listener
    if window:
        window.on_clear = brain.reset
    pump = asyncio.create_task(window.pump()) if window else None

    async def hear() -> np.ndarray | None:
        if window is None:
            return await listener.next_utterance()
        if window.down:  # key already held (e.g. pressed while Skelly was talking)
            window.pressed.set()
        else:
            window.set_state("idle")
        await window.pressed.wait()
        listener.begin()
        window.set_state("listening")
        await window.released.wait()
        return listener.end()

    with listener.stream:
        print("Skelly is listening. Ctrl-C to stop.", flush=True)
        while not (window and window.closed.is_set()):
            utterance = await hear()
            if utterance is None:
                continue
            listener.mute()
            try:
                if window:
                    window.set_state("thinking", heard="", said="")
                heard = await asyncio.to_thread(transcribe, utterance, config["whisper_model"])
                if not heard:
                    print("kid:    (nothing understood)", flush=True)
                    if window:
                        window.set_state("idle", heard="I didn't catch that. Hold SPACE and speak up!")
                        await asyncio.sleep(1.5)
                    continue
                print(f"kid:    {heard}", flush=True)
                if window:
                    window.set_state("thinking", heard=f"You: {heard}")
                async with session.lock:
                    await session.ensure_connected()
                    async for sentence in brain.reply(heard):
                        print(f"skelly: {sentence}", flush=True)
                        if window:
                            window.set_state("talking", said=f"Skelly: {sentence}")
                        await speak(sentence, config, session.device)
            except Exception as exc:
                print(f"turn failed: {exc!r}", flush=True)
                if window:
                    window.set_state("idle", said=f"Error: {exc}"[:300])
                    await asyncio.sleep(3)
            finally:
                await asyncio.sleep(0.3)
                listener.unmute()
    if pump:
        pump.cancel()


def listen_test(config: dict) -> int:
    """Print per-second mic RMS so the VAD threshold can be set above room noise."""
    print(f"threshold={config['vad_threshold']}; speak at Skelly distance for 5 s", flush=True)
    end = time.monotonic() + 5
    rate = int(sd.query_devices(config.get("mic_device"), "input")["default_samplerate"])
    with sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=config.get("mic_device")) as s:
        while time.monotonic() < end:
            block, _ = s.read(rate // 4)
            rms = float(np.sqrt(np.mean(block**2)))
            print(f"rms={rms:.4f} {'SPEECH' if rms > config['vad_threshold'] else ''}", flush=True)
    return 0
