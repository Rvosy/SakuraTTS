"""CPU G2PW session for local official models, without Torch or Transformers.

Prediction and sentence dedup follow GPT-SoVITS commit
48b1a0169a28582a8984402f82cf438d3bfa6aca, text/g2pw/onnx_api.py.
Original credits retained from that source:
https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw
https://github.com/GitYCC/g2pW
The related PaddlePaddle G2PW input helpers retain Apache-2.0 notices; the full
text is in docs/third-party/Apache-2.0.txt. The fixed GPT-SoVITS checkout also
supplies the MIT project license in docs/third-party/GPT-SoVITS-LICENSE.txt.

The default retains the official >510-token dedup boundary: grouping by original
text can share a token row despite query-specific truncation. See
research/experiments/2026-09-19-g2pw-dedup-boundary.md. No input semantics are repaired
implicitly here.
"""

import os
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Dict, List, Tuple

import numpy as np


class G2PWSession:
    def __init__(self, model_path, labels, *, sentence_dedup=True):
        self._initialize(model_path, labels, sentence_dedup=sentence_dedup, mapped_ort=False)

    @classmethod
    def from_ort_package(cls, package, labels, *, sentence_dedup=True):
        """Load a CPU graph prepared for this ORT build and machine.

        Initializers reference the mapped model file. Keep that file unchanged
        while the session is alive; the original ONNX loader remains available.
        File validation is streamed and included in this constructor's cost.
        """
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        import onnxruntime as ort

        package = Path(package)
        manifest = json.loads((package / "manifest.json").read_text())
        if manifest["format"] != "sakuratts-g2pw-ort-candidate-v1" or manifest["status"] != "converted_unvalidated":
            raise ValueError("Expected a prepared G2PW CPU ORT package")
        runtime = manifest["runtime"]
        if (runtime["version"] != ort.__version__ or runtime["build"] != ort.get_build_info()
                or runtime["system"] != platform.system() or runtime["machine"] != platform.machine()
                or runtime["providers"] != ["CPUExecutionProvider"]):
            raise ValueError("G2PW ORT package requires its original CPU runtime and architecture")
        model_path = package / manifest["model_file"]
        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != manifest["model_sha256"]:
            raise ValueError("G2PW ORT model hash differs from its package")
        runner = cls.__new__(cls)
        runner._initialize(model_path, labels, sentence_dedup=sentence_dedup, mapped_ort=True)
        return runner

    def _initialize(self, model_path, labels, *, sentence_dedup, mapped_ort):
        # Must precede ORT import for the current POSIX 1.30.0 telemetry lifecycle.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        import onnxruntime as ort

        self.model_path = Path(model_path).resolve()
        self.labels = list(labels)
        self.sentence_dedup = sentence_dedup
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        if mapped_ort:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.add_session_config_entry("session.use_memory_mapped_ort_model", "1")
            options.add_session_config_entry("session.use_ort_model_bytes_for_initializers", "1")
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])

    def run(self, model_input):
        """Return raw probabilities for exactly the supplied query tensors."""
        return self.session.run([], {
            "input_ids": model_input["input_ids"],
            "token_type_ids": model_input["token_type_ids"],
            "attention_mask": model_input["attention_masks"],
            "phoneme_mask": model_input["phoneme_masks"],
            "char_ids": model_input["char_ids"],
            "position_ids": model_input["position_ids"],
        })[0]

    def _predict(self, model_input):
        probabilities = self.run(model_input)
        indices = np.argmax(probabilities, axis=1).tolist()
        confidences = [row[index] for index, row in zip(indices, probabilities.tolist())]
        return [self.labels[index] for index in indices], confidences

    def predict(self, model_input, texts):
        """Return labels and confidences with the official dedup default."""
        if self.sentence_dedup:
            return self._predict_with_sentence_dedup(model_input, texts)
        return self._predict(model_input)

    def _predict_with_sentence_dedup(
        self, model_input: Dict[str, Any], texts: List[str]
    ) -> Tuple[List[str], List[float]]:
        if len(texts) <= 1:
            return self._predict(model_input=model_input)

        grouped_indices: Dict[str, List[int]] = {}
        for idx, text in enumerate(texts):
            grouped_indices.setdefault(text, []).append(idx)

        if all(len(indices) == 1 for indices in grouped_indices.values()):
            return self._predict(model_input=model_input)

        preds: List[str] = [""] * len(texts)
        confidences: List[float] = [0.0] * len(texts)
        for indices in grouped_indices.values():
            group_input = {name: value[indices] for name, value in model_input.items()}
            if len(indices) > 1:
                for name in ("input_ids", "token_type_ids", "attention_masks"):
                    group_input[name] = group_input[name][:1]

            group_preds, group_confidences = self._predict(model_input=group_input)
            for output_idx, pred, confidence in zip(indices, group_preds, group_confidences):
                preds[output_idx] = pred
                confidences[output_idx] = confidence

        return preds, confidences

    def close(self):
        """Release this session; process-wide ORT resources have their own lifetime."""
        self.session = None
