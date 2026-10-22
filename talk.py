"""Kids talk to Skelly; Whisper hears them, Claude answers, the jaw moves."""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from typing import AsyncIterator

import numpy as np
import sounddevice as sd

MIC_RATE = 16000
BLOCK = 480  # 30 ms
MODEL = "claude-opus-5-5"
SYSTEM = (
    "You are Skelly, a friendly 12-foot-tall skeleton standing in a front yard on Halloween night, "
    "chatting with trick-or-treaters, mostly young kids. Be playful and a little spooky, never actually "
    "frightening, mean, or gross. Everything you say is spoken aloud through your jaw, so answer in one "
    "or two short sentences a child can follow: no lists, no emoji, no stage directions. Ask a question "
    "back now and then. If a kid tells you their name or costume, use it. If asked something you should "
    "not answer, deflect with a bone pun."
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


class Listener:
    def __init__(self, threshold: float, device: int | str | None = None) -> None:
        self.segmenter = Segmenter(threshold)
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.muted = False
        self.loop = asyncio.get_running_loop()
        self.stream = sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                                     device=device, callback=self._callback)

    def _callback(self, indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        if not self.muted:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, indata[:, 0].copy())

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
                return utterance


def transcribe(audio: np.ndarray, model: str) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="en", fp16=True,
                                    condition_on_previous_text=False, initial_prompt="Talking to Skelly the skeleton.")
    return str(result["text"]).strip()


class Brain:
    def __init__(self, model: str = MODEL) -> None:
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic()
        self.model = model
        self.messages: list[dict] = []

    async def reply(self, text: str) -> AsyncIterator[str]:
        """Stream the answer back one sentence at a time."""
        self.messages.append({"role": "user", "content": text})
        buffer = ""
        async with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=300,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
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


async def run(config: dict, device: int) -> None:
    from skelly_mac import speak

    from skelly_mac import synthesize

    print("Loading speech models (first run downloads them)…", flush=True)
    await asyncio.to_thread(transcribe, np.zeros(MIC_RATE, dtype=np.float32), config["whisper_model"])
    await asyncio.to_thread(synthesize, "Boo.", config)
    brain = Brain(config.get("model", MODEL))
    listener = Listener(config["vad_threshold"], config.get("mic_device"))
    with listener.stream:
        print("Skelly is listening. Ctrl-C to stop.", flush=True)
        while True:
            utterance = await listener.next_utterance()
            listener.mute()
            try:
                heard = await asyncio.to_thread(transcribe, utterance, config["whisper_model"])
                if not heard:
                    continue
                print(f"kid:    {heard}", flush=True)
                async for sentence in brain.reply(heard):
                    print(f"skelly: {sentence}", flush=True)
                    await speak(sentence, config, device)
            except Exception as exc:
                print(f"turn failed: {exc}", flush=True)
            finally:
                await asyncio.sleep(0.3)
                listener.unmute()


def listen_test(config: dict) -> int:
    """Print per-second mic RMS so the VAD threshold can be set above room noise."""
    print(f"threshold={config['vad_threshold']}; speak at Skelly distance for 5 s", flush=True)
    end = time.monotonic() + 5
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32", device=config.get("mic_device")) as s:
        while time.monotonic() < end:
            block, _ = s.read(MIC_RATE // 4)
            rms = float(np.sqrt(np.mean(block**2)))
            print(f"rms={rms:.4f} {'SPEECH' if rms > config['vad_threshold'] else ''}", flush=True)
    return 0
