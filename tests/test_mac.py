import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

import skelly_mac
import talk


class SegmenterTests(unittest.TestCase):
    def blocks(self, seconds: float, level: float) -> list[np.ndarray]:
        n = int(seconds * talk.MIC_RATE / talk.BLOCK)
        rng = np.random.default_rng(0)
        return [rng.normal(0, level, talk.BLOCK).astype(np.float32) for _ in range(n)]

    def test_speech_between_silence_yields_one_utterance(self):
        seg = talk.Segmenter(threshold=0.02)
        out = [seg.feed(b) for b in self.blocks(1, 0.001) + self.blocks(1, 0.1) + self.blocks(1.5, 0.001)]
        utterances = [u for u in out if u is not None]
        self.assertEqual(len(utterances), 1)
        self.assertGreater(len(utterances[0]), talk.MIC_RATE)  # ~1 s of speech plus pre-roll and tail

    def test_short_blip_is_dropped(self):
        seg = talk.Segmenter(threshold=0.02)
        out = [seg.feed(b) for b in self.blocks(0.5, 0.001) + self.blocks(0.15, 0.1) + self.blocks(1.5, 0.001)]
        self.assertTrue(all(u is None for u in out))


class SentenceTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(talk.split_sentences("Boo! Who goes there? I am"), (["Boo!", "Who goes there?"], "I am"))
        self.assertEqual(talk.split_sentences("No boundary yet"), ([], "No boundary yet"))


class ConfigTests(unittest.TestCase):
    def test_paired_live_mac(self):
        output = (
            'address: 74-77-86-37-de-4a, not connected, paired, name: "AirPods", recent access date: x\n'
            'address: 13-c0-02-6e-21-3c, not connected, paired, name: "12ft Skelly(Live)", recent access date: x\n'
        )
        self.assertEqual(skelly_mac.paired_live_mac(output), "13:C0:02:6E:21:3C")
        self.assertIsNone(skelly_mac.paired_live_mac("address: 00-00-00-00-00-00, paired, name: \"12ft Skelly\"\n"))



if __name__ == "__main__":
    unittest.main()


class PushToTalkTests(unittest.TestCase):
    def test_begin_end_collects_blocks_and_drops_short_presses(self):
        import asyncio

        async def go():
            listener = talk.Listener.__new__(talk.Listener)
            listener.rate = talk.MIC_RATE
            listener.muted = False
            listener.recording = None
            listener.queue = asyncio.Queue()
            listener.loop = asyncio.get_running_loop()
            listener.silero, listener.armed, listener.speech_prob = None, False, 0.5
            block = np.zeros((talk.BLOCK, 1), dtype=np.float32)
            listener.begin()
            for _ in range(3):
                listener._callback(block, talk.BLOCK, None, None)
            self.assertIsNone(listener.end())  # 90 ms is a bump, not a word
            listener.begin()
            for _ in range(20):
                listener._callback(block, talk.BLOCK, None, None)
            self.assertEqual(len(listener.end()), 20 * talk.BLOCK)
            self.assertIsNone(listener.recording)

        asyncio.run(go())


class ResampleTests(unittest.TestCase):
    def test_48k_mic_audio_reaches_whisper_at_16k(self):
        tone = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
        out = talk.to_whisper_rate(tone, 48000)
        self.assertAlmostEqual(len(out), talk.MIC_RATE, delta=20)
        self.assertIs(talk.to_whisper_rate(tone, talk.MIC_RATE), tone)


class MemoryTests(unittest.TestCase):
    def test_reset_forgets_everything(self):
        brain = talk.Brain.__new__(talk.Brain)
        brain.messages = [{"role": "user", "content": "I am Olivia"}, {"role": "assistant", "content": "Hello Olivia"}]
        brain.reset()
        self.assertEqual(brain.messages, [])


class VoiceFxTests(unittest.TestCase):
    def dominant_hz(self, pcm: np.ndarray) -> float:
        mono = pcm[:, 0].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(mono))
        return float(np.fft.rfftfreq(len(mono), 1 / skelly_mac.RATE)[np.argmax(spectrum)])

    def test_pitch_down_is_relative_to_output_rate_for_24k_tts(self):
        rate = 24000
        tone = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(rate * 2) / rate)).astype(np.float32)
        plain = skelly_mac.decode_audio(tone.tobytes(), None, rate)
        pitched = skelly_mac.decode_audio(tone.tobytes(), "asetrate=44100*0.9,aresample=44100", rate)
        self.assertAlmostEqual(self.dominant_hz(plain), 1000, delta=5)
        self.assertAlmostEqual(self.dominant_hz(pitched), 900, delta=5)  # down, never a chipmunk
        tail = skelly_mac.TAIL_SECONDS * skelly_mac.RATE
        self.assertAlmostEqual((len(pitched) - tail) / (len(plain) - tail), 1 / 0.9, delta=0.02)

    def test_every_clip_ends_with_silence(self):
        rate = 24000
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
        pcm = skelly_mac.decode_audio(tone.tobytes(), None, rate)
        self.assertAlmostEqual(len(pcm) / skelly_mac.RATE, 1 + skelly_mac.TAIL_SECONDS, delta=0.05)
        self.assertEqual(int(np.abs(pcm[-int(0.5 * skelly_mac.RATE):]).max()), 0)


class HandsFreeTests(unittest.TestCase):
    def test_plausible_filters_whisper_noise(self):
        self.assertTrue(talk.plausible("Hi Skelly, I am a dragon!"))
        self.assertTrue(talk.plausible("Trick or treat"))
        self.assertTrue(talk.plausible("boo"))
        self.assertFalse(talk.plausible("Thank you."))
        self.assertFalse(talk.plausible("Thanks for watching!"))
        self.assertFalse(talk.plausible("the"))
        self.assertFalse(talk.plausible(""))

    def test_callback_queues_only_in_handsfree(self):
        import asyncio

        async def go():
            listener = talk.Listener.__new__(talk.Listener)
            listener.rate, listener.muted, listener.recording, listener.handsfree = talk.MIC_RATE, False, None, False
            listener.queue, listener.loop = asyncio.Queue(), asyncio.get_running_loop()
            listener.silero, listener.armed, listener.speech_prob = None, False, 0.5
            listener.segmenter = talk.Segmenter(0.02)
            quiet = np.full((talk.BLOCK, 1), 0.001, dtype=np.float32)
            loud = np.full((talk.BLOCK, 1), 0.1, dtype=np.float32)
            utterance = [quiet] * 5 + [loud] * 20 + [quiet] * 40

            for block in utterance:
                listener._callback(block, talk.BLOCK, None, None)
            await asyncio.sleep(0)
            self.assertTrue(listener.queue.empty())  # push-to-talk: the VAD is idle
            self.assertFalse(listener.segmenter.active)
            self.assertAlmostEqual(listener.level, 0.001, places=3)
            listener.set_handsfree(True)
            for block in utterance:
                listener._callback(block, talk.BLOCK, None, None)
            await asyncio.sleep(0)
            self.assertEqual(listener.queue.qsize(), 1)  # one finished utterance, not raw blocks

        asyncio.run(go())


class DefaultOutputTests(unittest.TestCase):
    def test_restore_never_points_the_mac_at_the_skelly(self):
        calls = []
        original = skelly_mac.run
        skelly_mac.run = lambda *a, **k: calls.append(a) or ""
        try:
            skelly_mac.restore_default_outputs({"output": "MacBook Pro Speakers", "system": "12ft Skelly(Live)"})
        finally:
            skelly_mac.run = original
        self.assertEqual(calls, [("SwitchAudioSource", "-s", "MacBook Pro Speakers", "-t", "output")])


class StreamJobTests(unittest.TestCase):
    def test_chunks_flow_through_fx_to_frame_aligned_output(self):
        import asyncio

        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(skelly_mac.STREAM_RATE) / skelly_mac.STREAM_RATE)).astype(np.float32)
        chunks = [tone[i:i + 6000].tobytes() for i in range(0, len(tone), 6000)]
        original = skelly_mac.synthesize_stream
        skelly_mac.synthesize_stream = lambda text, config: iter(chunks)
        try:
            async def go():
                job = skelly_mac.StreamJob("x", {"voice_fx": "volume=2dB", "tts": "clone"})
                await asyncio.wait_for(job.started, 10)
                frames = await asyncio.to_thread(lambda: list(job.frames()))
                await job.generation
                return frames

            frames = asyncio.run(go())
        finally:
            skelly_mac.synthesize_stream = original
        self.assertTrue(all(len(f) % 4 == 0 for f in frames))
        seconds = sum(len(f) for f in frames) / 4 / skelly_mac.RATE
        self.assertAlmostEqual(seconds, 1 + skelly_mac.TAIL_SECONDS, delta=0.05)


class FillerTests(unittest.TestCase):
    def test_filler_has_no_silence_pad(self):
        rate = 24000
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
        pcm = skelly_mac.decode_audio(tone.tobytes(), None, rate, pad=False)
        self.assertAlmostEqual(len(pcm) / skelly_mac.RATE, 1.0, delta=0.05)


class InterruptTests(unittest.TestCase):
    def test_interrupted_reply_keeps_what_was_said(self):
        import asyncio

        async def go():
            brain = talk.Brain.__new__(talk.Brain)
            brain.messages = []

            async def fake_stream(_buffer):
                yield "Boo!"
                await asyncio.sleep(10)
                yield "never"

            brain._stream = fake_stream
            heard = []

            async def consume():
                async for sentence in brain.reply("hi"):
                    heard.append(sentence)

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return heard, brain.messages

        heard, messages = asyncio.run(go())
        self.assertEqual(heard, ["Boo!"])
        self.assertEqual(messages, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Boo!"}])

    def test_stream_job_stop_ends_generation_and_frames(self):
        import asyncio, time

        def slow_chunks(text, config):
            tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(12000) / 24000)).astype(np.float32).tobytes()
            for _ in range(40):
                yield tone
                time.sleep(0.05)

        original = skelly_mac.synthesize_stream
        skelly_mac.synthesize_stream = slow_chunks
        try:
            async def go():
                job = skelly_mac.StreamJob("x", {"voice_fx": None, "tts": "clone"})
                await asyncio.wait_for(job.started, 10)
                reader = asyncio.to_thread(lambda: sum(len(f) for f in job.frames()))
                await asyncio.sleep(0.2)
                job.stop()
                got = await asyncio.wait_for(reader, 5)
                await asyncio.wait_for(job.generation, 5)  # must not raise after a stop
                return got

            got = asyncio.run(go())
        finally:
            skelly_mac.synthesize_stream = original
        self.assertLess(got / 4 / skelly_mac.RATE, 10)  # far less than the 20 s it would have produced


class BargeInTests(unittest.TestCase):
    def test_threshold_sits_between_bleed_and_voice_or_is_off(self):
        self.assertAlmostEqual(talk.pick_barge_threshold(0.004, 0.064), 0.016, places=4)
        self.assertIsNone(talk.pick_barge_threshold(0.02, 0.03))
        self.assertIsNone(talk.pick_barge_threshold(0.0, 0.03))

    def test_voice_over_bleed_sets_barge_in_and_keeps_the_utterance(self):
        import asyncio

        async def go():
            listener = talk.Listener.__new__(talk.Listener)
            listener.rate, listener.muted, listener.recording, listener.handsfree = talk.MIC_RATE, True, None, True
            listener.queue, listener.loop = asyncio.Queue(), asyncio.get_running_loop()
            listener.silero, listener.armed, listener.speech_prob = None, False, 0.5
            listener.segmenter = talk.Segmenter(0.02)
            listener.vad_threshold = 0.02
            listener.barge_in = asyncio.Event()
            listener.segmenter.on_start = lambda: listener.loop.call_soon_threadsafe(listener.barge_in.set)
            listener.arm_barge_in(0.05)
            quiet = np.full((talk.BLOCK, 1), 0.03, dtype=np.float32)  # Skelly's bleed: under the barge threshold
            loud = np.full((talk.BLOCK, 1), 0.2, dtype=np.float32)
            for _ in range(10):
                listener._callback(quiet, talk.BLOCK, None, None)
            await asyncio.sleep(0)
            self.assertFalse(listener.barge_in.is_set())
            for _ in range(6):
                listener._callback(loud, talk.BLOCK, None, None)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertTrue(listener.barge_in.is_set())
            self.assertTrue(listener.segmenter.active)
            listener.disarm_barge_in()
            listener.unmute()  # not muted -> must NOT reset the utterance in progress
            self.assertTrue(listener.segmenter.active)
            self.assertEqual(listener.segmenter.threshold, 0.02)

        asyncio.run(go())


class RefusalTests(unittest.TestCase):
    def _brain(self, finals):
        import asyncio

        brain = talk.Brain.__new__(talk.Brain)
        brain.messages, brain.model, brain.persona = [], "claude-sonnet-5-5", "p"
        calls = []

        class Stream:
            def __init__(self, text, final):
                self.text, self.final = text, final

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            @property
            def text_stream(self):
                async def gen():
                    yield self.text

                return gen()

            async def get_final_message(self):
                return self.final

        class Messages:
            def stream(self, **kwargs):
                calls.append(kwargs)
                text, final = finals[len(calls) - 1]
                return Stream(text, final)

        brain.client = type("C", (), {"beta": type("B", (), {"messages": Messages()})()})()
        return brain, calls

    def _final(self, stop_reason, text=True):
        block = type("T", (), {"type": "text", "text": "x"})()
        details = type("D", (), {"category": "general_harms"})()
        return type("F", (), {"stop_reason": stop_reason, "content": [block] if text else [], "stop_details": details})()

    def test_refusal_before_any_sentence_retries_gently_then_answers(self):
        import asyncio

        brain, calls = self._brain([("I will eat", self._final("refusal", text=False)), ("Boo, silly goose!", self._final("end_turn"))])
        said = asyncio.run(self._collect(brain))
        self.assertEqual(said, ["Boo, silly goose!"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["messages"][-1]["role"], "system")  # the gentle nudge
        self.assertEqual([m["role"] for m in brain.messages], ["user", "assistant"])

    def test_refusal_after_a_full_sentence_keeps_it_and_adds_nothing(self):
        import asyncio

        brain, calls = self._brain([("Hello there! I will now eat", self._final("refusal"))])
        said = asyncio.run(self._collect(brain))
        self.assertEqual(said, ["Hello there!"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(brain.messages[-1], {"role": "assistant", "content": "Hello there!"})

    def test_double_refusal_gives_the_excuse(self):
        import asyncio

        brain, _ = self._brain([("", self._final("refusal", text=False)), ("", self._final("refusal", text=False))])
        said = asyncio.run(self._collect(brain))
        self.assertEqual(said, ["Hmm, that one rattled my skull. Ask me something else!"])
        self.assertEqual(brain.messages, [])

    async def _collect(self, brain):
        return [s async for s in brain.reply("hi")]


class JawTestTests(unittest.TestCase):
    def test_staircase_levels_and_burst_rates(self):
        import subprocess, tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        skelly_mac.jaw_test_wav(path)
        raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-"],
                             capture_output=True, check=True).stdout
        path.unlink()
        a = np.frombuffer(raw, dtype=np.float32)
        peak_db = 20 * np.log10(np.abs(a).max())
        self.assertAlmostEqual(peak_db, -3, delta=0.2)  # loudest step is exactly -3 dB
        self.assertGreater(len(a) / skelly_mac.RATE, 15)  # staircase + four burst trains


class ArticulatorTests(unittest.TestCase):
    def test_valleys_are_ducked_and_streaming_matches_whole(self):
        rate = skelly_mac.RATE
        t = np.arange(rate) / rate
        env = np.where((t % 0.25) < 0.15, 1.0, 0.3)  # 4 syllables/s: loud 150 ms, soft 100 ms
        mono = (0.5 * env * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        pcm = np.column_stack([mono, mono])
        whole = skelly_mac.Articulator(0.5).process(pcm, final=True)
        soft = whole[int(0.2 * rate):int(0.24 * rate), 0]
        loud = whole[int(0.05 * rate):int(0.1 * rate), 0]
        self.assertLess(np.abs(soft).max(), 0.05 * 32767 * 0.5 * 0.3 + 1)  # ducked ~30 dB
        self.assertGreater(np.abs(loud).max(), 0.4 * 32767)  # untouched
        streamer = skelly_mac.Articulator(0.5)
        pieces = [streamer.process(pcm[i:i + 1000]) for i in range(0, len(pcm), 1000)] + [streamer.process(pcm[:0], final=True)]
        np.testing.assert_array_equal(np.concatenate(pieces), whole)
        self.assertEqual(len(skelly_mac.Articulator(0).process(pcm, final=True)), len(pcm))


class SaveConfigTests(unittest.TestCase):
    def test_defaults_are_not_baked_into_the_file(self):
        import json, tempfile

        original = skelly_mac.CONFIG_FILE
        with tempfile.TemporaryDirectory() as d:
            skelly_mac.CONFIG_FILE = Path(d) / "mac.json"
            try:
                skelly_mac.save_config({**skelly_mac.DEFAULTS, "mic_device": "Yeti", "live_mac": "AA"})
                saved = json.loads(skelly_mac.CONFIG_FILE.read_text())
            finally:
                skelly_mac.CONFIG_FILE = original
        self.assertEqual(saved, {"mic_device": "Yeti", "live_mac": "AA"})
        self.assertNotIn("voice_fx", saved)


class SileroTests(unittest.TestCase):
    def test_segmenter_takes_an_external_speech_decision(self):
        seg = talk.Segmenter(threshold=0.02)
        loud_but_not_voice = np.full(talk.BLOCK, 0.2, dtype=np.float32)
        self.assertTrue(all(seg.feed(loud_but_not_voice, loud=False) is None for _ in range(20)))
        self.assertFalse(seg.active)  # a loud car is not speech
        for _ in range(5):
            seg.feed(loud_but_not_voice, loud=True)
        self.assertTrue(seg.active)

    def test_silero_resamples_and_scores(self):
        vad = talk.SileroVAD()
        rate = 48000
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.05, rate).astype(np.float32)
        self.assertLess(vad.feed(noise, rate), 0.2)
        vad.reset()
        self.assertEqual(len(vad.buffer), 0)

    def test_callback_uses_voice_not_level_and_level_gate_when_armed(self):
        import asyncio

        async def go():
            listener = talk.Listener.__new__(talk.Listener)
            listener.rate, listener.muted, listener.recording, listener.handsfree = talk.MIC_RATE, False, None, True
            listener.queue, listener.loop = asyncio.Queue(), asyncio.get_running_loop()
            listener.silero, listener.armed, listener.speech_prob = None, False, 0.5
            listener.segmenter = talk.Segmenter(0.02)
            listener.vad_threshold, listener.speech_prob, listener.armed = 0.02, 0.5, False
            listener.barge_in = asyncio.Event()
            listener.segmenter.on_start = lambda: None
            fake = type("F", (), {"prob": 0.0, "feed": lambda self, block, rate: self.prob})()
            listener.silero = fake
            loud = np.full((talk.BLOCK, 1), 0.2, dtype=np.float32)
            quiet_voice = np.full((talk.BLOCK, 1), 0.01, dtype=np.float32)
            fake.prob = 0.1
            for _ in range(10):
                listener._callback(loud, talk.BLOCK, None, None)
            self.assertFalse(listener.segmenter.active)  # loud non-voice ignored
            fake.prob = 0.95
            for _ in range(5):
                listener._callback(quiet_voice, talk.BLOCK, None, None)
            self.assertTrue(listener.segmenter.active)  # quiet voice counts
            listener.segmenter.reset()
            listener.arm_barge_in(0.1)
            for _ in range(10):
                listener._callback(quiet_voice, talk.BLOCK, None, None)
            self.assertFalse(listener.segmenter.active)  # Skelly's own voice at the mic stays under the gate
            for _ in range(6):
                listener._callback(loud, talk.BLOCK, None, None)
            self.assertTrue(listener.segmenter.active)  # a kid over the bleed

        asyncio.run(go())
