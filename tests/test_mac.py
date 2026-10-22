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

    def test_say_device(self):
        listing = "  100 MacBook Pro Speakers\n  118 12ft Skelly(Live)\n   93 ZoomAudioDevice\n"
        self.assertEqual(skelly_mac.say_device(listing), "118")
        with self.assertRaises(RuntimeError):
            skelly_mac.say_device("  100 MacBook Pro Speakers\n")


if __name__ == "__main__":
    unittest.main()
