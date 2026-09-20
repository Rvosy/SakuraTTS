import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/tools"))
from trace_reference import ReferenceTrace


class TraceReferenceTests(unittest.TestCase):
    def test_sampling_noise_capture_preserves_draws_and_rng_state(self):
        module = ModuleType("AR.models.t2s_model")
        utils = ModuleType("AR.models.utils")

        def draw(probabilities):
            noise = torch.empty_like(probabilities).exponential_(1)
            return torch.argmax(probabilities / noise, dim=-1, keepdim=True).to(torch.int)

        def sample(logits, previous_tokens=None):
            probabilities = torch.softmax(logits, dim=-1)
            return utils.multinomial_sample_one_no_sync(probabilities), probabilities

        utils.multinomial_sample_one_no_sync = draw
        module.sample = sample
        noop = lambda *a, **kw: None
        gpt = SimpleNamespace(
            EOS=1024, infer_panel_naive=noop, ar_predict_layer=torch.nn.Identity(),
            t2s_transformer=SimpleNamespace(process_prompt=noop, decode_next_token=noop),
        )
        engine = SimpleNamespace(
            text_preprocessor=SimpleNamespace(pre_seg_text=noop, clean_text_inf=noop,
                                              segment_and_extract_feature_for_text=noop),
            _set_prompt_semantic=noop, _get_ref_spec=noop, audio_postprocess=noop,
            t2s_model=SimpleNamespace(model=gpt), vits_model=SimpleNamespace(decode=noop),
        )
        inputs = [torch.arange(size, dtype=torch.float32).reshape(1, -1) / size for size in (1024, 1025)]
        torch.manual_seed(931)
        expected = [sample(value)[0] for value in inputs]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(931)
        with patch.dict(sys.modules, {module.__name__: module, utils.__name__: utils}):
            trace = ReferenceTrace(engine, "official", noop, capture_sampling_noise=True)
            try:
                for index, logits in enumerate(inputs):
                    token, probabilities = module.sample(logits)
                    self.assertTrue(torch.equal(token, expected[index]))
                    noise = torch.from_numpy(trace.arrays[f"sampling_noise.{index}"])
                    replay = torch.argmax(probabilities / noise, dim=-1, keepdim=True).to(torch.int)
                    self.assertTrue(torch.equal(token, replay))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            finally:
                trace.close()
            self.assertIs(module.sample, sample)
            self.assertIs(utils.multinomial_sample_one_no_sync, draw)

    def test_stop_checks_logits_after_inplace_repetition_penalty(self):
        module = ModuleType("AR.models.t2s_model")

        def sample(logits, previous_tokens, **kwargs):
            logits[0, previous_tokens[0]] /= 1.35
            return torch.tensor([[0]]), None

        module.sample = sample
        noop = lambda *a, **kw: None
        gpt = SimpleNamespace(
            EOS=1, infer_panel_naive=noop, ar_predict_layer=torch.nn.Identity(),
            t2s_transformer=SimpleNamespace(process_prompt=noop, decode_next_token=noop),
        )
        engine = SimpleNamespace(
            text_preprocessor=SimpleNamespace(pre_seg_text=noop, clean_text_inf=noop,
                                              segment_and_extract_feature_for_text=noop),
            _set_prompt_semantic=noop, _get_ref_spec=noop, audio_postprocess=noop,
            t2s_model=SimpleNamespace(model=gpt), vits_model=SimpleNamespace(decode=noop),
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            trace = ReferenceTrace(engine, "official", noop)
            try:
                module.sample(torch.tensor([[10.0, 9.0]]), torch.tensor([[0]]))
                recorded = trace.samples[-1]
                self.assertEqual(recorded["token"], 0)
                self.assertEqual(recorded["argmax_before_sampling"], 0)
                self.assertEqual(recorded["argmax_after_sampling"], 1)
                self.assertEqual(trace.stop_reason(recorded), "argmax_eos")
            finally:
                trace.close()
            self.assertIs(module.sample, sample)


if __name__ == "__main__":
    unittest.main()
