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
            listener.muted = False
            listener.recording = None
            listener.queue = asyncio.Queue()
            listener.loop = asyncio.get_running_loop()
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
