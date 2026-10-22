"""Kids talk to Skelly; Whisper hears them, Claude answers, the jaw moves."""

from __future__ import annotations

import asyncio
import random
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
    def __init__(self, threshold: float, device: int | str | None = None, silence_seconds: float = 0.8) -> None:
        self.segmenter = Segmenter(threshold, silence_blocks=int(silence_seconds * 1000 / 30))
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.muted = False
        self.handsfree = False  # only then do blocks flow to the VAD queue
        self.level = 0.0  # last block RMS, for the meter
        self.loop = asyncio.get_running_loop()
        self.rate = int(sd.query_devices(device, "input")["default_samplerate"])
        self.stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="float32",
                                     blocksize=BLOCK * self.rate // MIC_RATE, device=device, callback=self._callback)

        self.recording: list[np.ndarray] | None = None

    def _callback(self, indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        mono = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(mono**2)))
        if self.recording is not None:
            self.recording.append(mono)
        elif self.handsfree and not self.muted:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, mono)

    def set_handsfree(self, enabled: bool) -> None:
        self.handsfree = enabled
        self.unmute()

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


SHORT_OK = {"hi", "hello", "boo", "hey", "yes", "no", "bye", "trick or treat", "happy halloween", "hi skelly",
            "hey skelly", "hello skelly", "who are you", "what are you"}
HALLUCINATIONS = {"thank you", "thanks for watching", "you", "bye", "thank you for watching", "subscribe"}


def plausible(text: str) -> bool:
    """Drop Whisper's noise hallucinations and stray fragments in hands-free mode."""
    words = re.sub(r"[^a-z' ]", "", text.lower()).split()
    phrase = " ".join(words)
    if not words or phrase in HALLUCINATIONS:
        return False
    return len(words) >= 3 or phrase in SHORT_OK


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
        extra = {}  # server-side refusal fallbacks exist for Opus/Sonnet/Fable, not Haiku
        if not self.model.startswith("claude-haiku"):
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        async with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=300,
            system=[{"type": "text", "text": self.persona, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
            output_config={"effort": "low"},
            **extra,
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
    from skelly_mac import FILLERS, FILLER_AFTER, MODELS, StreamJob, filler_pcm, play_pcm, render, synthesize

    config = session.config
    window = None if vad else __import__("ui").TalkWindow()
    print("Loading speech models (first run downloads them)…", flush=True)
    await asyncio.to_thread(transcribe, np.zeros(MIC_RATE, dtype=np.float32), config["whisper_model"])
    await asyncio.to_thread(synthesize, "Boo.", config)
    streaming = config["tts"] in ("clone", "qwen")
    fillers: list[np.ndarray] = []
    if config.get("filler"):
        print("Preparing fillers…", flush=True)
        fillers = [await asyncio.to_thread(filler_pcm, config, line) for line in FILLERS]
    recent: list[int] = []

    def pick_filler() -> np.ndarray:
        choice = random.choice([i for i in range(len(fillers)) if i not in recent])
        recent.append(choice)
        del recent[:-6]  # no repeats within the last six
        return fillers[choice]

    brain = Brain(config.get("model", MODEL), config.get("persona") or PERSONA)
    listener = Listener(config["vad_threshold"], config.get("mic_device"), float(config.get("vad_silence", 1.2)))
    listener.set_handsfree(vad)
    session.listener = listener
    if window:
        window.on_clear = brain.reset
        window.level_source = lambda: (listener.level, listener.segmenter.threshold)
        models = {**MODELS}
        if brain.model not in models.values():
            models[brain.model] = brain.model
        window.set_models(models, brain.model)

        def switch_model(model_id: str) -> None:
            brain.model = model_id
            print(f"model: {model_id}", flush=True)

        window.on_model = switch_model
    pump = asyncio.create_task(window.pump()) if window else None

    async def first(*coros: object) -> int:
        """Await several things; return the index of whichever finishes first and cancel the rest."""
        tasks = [asyncio.ensure_future(c) for c in coros]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        return tasks.index(next(iter(done)))

    async def hear() -> np.ndarray | None:
        if window is None:
            return await listener.next_utterance()
        while True:
            window.mode_changed.clear()
            if window.handsfree:
                listener.set_handsfree(True)
                window.set_state("waiting")
                vad_task = asyncio.ensure_future(listener.next_utterance())
                if await first(vad_task, window.mode_changed.wait()) == 0:
                    return vad_task.result()
                continue
            listener.set_handsfree(False)
            if window.down:  # key already held (e.g. pressed while Skelly was talking)
                window.pressed.set()
            else:
                window.set_state("idle")
            if await first(window.pressed.wait(), window.mode_changed.wait()) == 1:
                continue
            listener.begin()
            window.set_state("listening")
            await window.released.wait()
            return listener.end()

    async def speak_streamed(heard: str) -> dict[str, float]:
        """Render sentence N+1 while sentence N plays, so there is no dead air between them."""
        queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()
        timing = {"llm": 0.0, "tts": 0.0}
        start = time.monotonic()

        async def produce() -> None:
            try:
                async for sentence in brain.reply(heard):
                    timing["llm"] = timing["llm"] or time.monotonic() - start
                    print(f"skelly: {sentence}", flush=True)
                    job = StreamJob(sentence, config) if streaming else asyncio.create_task(render(sentence, config))
                    queue.put_nowait((sentence, job))
            finally:
                queue.put_nowait(None)

        producer = asyncio.create_task(produce())
        first = True
        while (item := await queue.get()) is not None:
            sentence, job = item
            if first:
                ready = job.started if streaming else job
                try:
                    await asyncio.wait_for(asyncio.shield(ready), FILLER_AFTER)
                except asyncio.TimeoutError:
                    if fillers:
                        print("skelly: (filler)", flush=True)
                        await asyncio.to_thread(play_pcm, pick_filler(), session.device)
                await ready
                timing["tts"] = time.monotonic() - start - timing["llm"]
                first = False
            if window:
                window.set_state("talking", said=f"Skelly: {sentence}")
            if streaming:
                await asyncio.to_thread(job.play, session.device)
                await job.generation
            else:
                await asyncio.to_thread(play_pcm, await job, session.device)
        await producer
        return timing

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
                t0 = time.monotonic()
                heard = await asyncio.to_thread(transcribe, utterance, config["whisper_model"])
                t_stt = time.monotonic() - t0
                handsfree = vad or (window is not None and window.handsfree)
                if not heard or (handsfree and not plausible(heard)):
                    print(f"kid:    (ignored: {heard!r})", flush=True)
                    if window and not handsfree:
                        window.set_state("idle", heard="I didn't catch that. Hold SPACE and speak up!")
                        await asyncio.sleep(1.5)
                    continue
                print(f"kid:    {heard}", flush=True)
                if window:
                    window.set_state("thinking", heard=f"You: {heard}", said="")
                async with session.lock:
                    await session.ensure_connected()
                    timing = await speak_streamed(heard)
                print(f"timing: stt {t_stt:.1f}s | first sentence from claude {timing['llm']:.1f}s | "
                      f"its tts {timing['tts']:.1f}s | first audio {t_stt + timing['llm'] + timing['tts']:.1f}s after release", flush=True)
            except Exception as exc:
                print(f"turn failed: {exc!r}", flush=True)
                if window:
                    window.set_state("idle", said=f"Error: {exc}"[:300])
                    await asyncio.sleep(3)
            finally:
                await asyncio.sleep(0.5)  # let the Skelly's last echo die before the mic re-arms
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
