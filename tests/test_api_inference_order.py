"""api_v2 singleton batches sort inference, then restore audible text order."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_nvidia_failure_lifecycle import engine, speech


class ApiInferenceOrderTests(unittest.TestCase):
    def run_request(self, *, split_bucket=None, streaming=False, replay=False):
        model = engine("resident")
        texts = ("a much longer sentence", "short", "equal")
        request = SimpleNamespace(seconds=0., fragments=tuple(
            SimpleNamespace(target={"norm_text": text, "phones": [index]})
            for index, text in enumerate(texts)))
        calls, emitted = [], []
        draws = np.random.default_rng(1234).integers(1, 30000, size=3)
        replay_inputs = [dict(draws=np.full((1, 2), index + 1.), noise=index + 10.)
                         for index in range(3)]

        def semantic(prepared, reference, *, rng, semantic_random_draw, **kwargs):
            index = prepared.target["phones"][0]
            calls.append(index)
            if replay:
                np.testing.assert_array_equal(semantic_random_draw(0, (1, 2)), [[index + 1.] * 2])
            return SimpleNamespace(index=index, draw=rng.integers(1, 30000), generation=speech().generation)

        def acoustic(generated, *, acoustic_noise, **kwargs):
            if replay:
                self.assertEqual(acoustic_noise, generated.index + 10.)
            result = speech()
            result.pcm = np.array([generated.draw], dtype=np.int16)
            return result

        with patch("sakuratts.backends.cuda.engine.prepare_text_request", return_value=request), \
                patch("sakuratts.backends.cuda.engine.generate_prepared_semantic", side_effect=semantic), \
                patch("sakuratts.backends.cuda.engine.synthesize_acoustic", side_effect=acoustic):
            options = {} if split_bucket is None else {"split_bucket": split_bucket}
            pcm, report = model.synthesize("original text", **options,
                on_fragment=(lambda pcm, rate: emitted.extend(pcm.tolist())) if streaming else None,
                collect_audio=not streaming, random_inputs=replay_inputs if replay else None)
        self.assertFalse(model.busy)
        return pcm, report, calls, emitted, draws

    def test_bucketed_singleton_batches_restore_audio_and_reports(self):
        pcm, report, calls, _, draws = self.run_request(split_bucket=True, replay=True)
        # Equal normalized lengths keep their original relative order.
        self.assertEqual(calls, [1, 2, 0])
        np.testing.assert_array_equal(pcm, draws[[2, 0, 1]])
        self.assertEqual(report["execution_order"], calls)
        self.assertEqual([part["index"] for part in report["fragments"]], [0, 1, 2])
        self.assertTrue(report["parameters"]["split_bucket"])

    def test_low_level_default_preserves_original_order(self):
        pcm, report, calls, _, draws = self.run_request()
        self.assertEqual(calls, [0, 1, 2])
        np.testing.assert_array_equal(pcm, draws)
        self.assertFalse(report["parameters"]["split_bucket"])

    def test_streaming_disables_buckets_and_emits_in_text_order(self):
        pcm, report, calls, emitted, draws = self.run_request(split_bucket=True, streaming=True)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(pcm.size, 0)
        np.testing.assert_array_equal(emitted, draws)
        self.assertFalse(report["parameters"]["split_bucket"])


if __name__ == "__main__":
    unittest.main()
