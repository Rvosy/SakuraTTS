"""Local G2PW pinyin path for the validated official 1.1 model configuration.

Composes the independently checked text, tokenizer, input and CPU session
modules. Call order follows GPT-SoVITS 48b1a0169a28582a8984402f82cf438d3bfa6aca,
text/g2pw/onnx_api.py (_G2PWBaseOnnxConverter.__call__). That source credits
PaddleSpeech and GitYCC/g2pW; see g2pw_text.py and docs/third-party for notices.
The current model uses phoneme labels (use_char_phoneme=False) and masking.
This is not Chinese segmentation, tone sandhi, phone conversion or TTS.
"""

from pathlib import Path

from sakuratts.frontend.g2pw_inputs import G2PWInputs
from sakuratts.frontend.g2pw_session import G2PWSession
from sakuratts.frontend.g2pw_text import G2PWText
from sakuratts.frontend.tokenizer import ChineseBertTokenizer


class G2PW:
    def __init__(self, resource_dir, tokenizer_json, model_path=None, *, context_chars=16,
                 enable_opencc=True, sentence_dedup=True, ort_package=None):
        if (model_path is None) == (ort_package is None):
            raise ValueError("Provide an ONNX model_path or a prepared ort_package")
        self.text = G2PWText(resource_dir, context_chars=context_chars, enable_opencc=enable_opencc)
        # Model IDs use the complete table, before text-query exclusions.
        polyphonic = [line.split("\t") for line in
                      (Path(resource_dir) / "POLYPHONIC_CHARS.txt").read_text().strip().splitlines()]
        self.inputs = G2PWInputs(ChineseBertTokenizer(tokenizer_json), polyphonic)
        self.session = (G2PWSession(model_path, self.inputs.labels, sentence_dedup=sentence_dedup)
                        if ort_package is None else
                        G2PWSession.from_ort_package(ort_package, self.inputs.labels, sentence_dedup=sentence_dedup))

    def __call__(self, sentences):
        texts, model_positions, result_positions, sentence_ids, partials = self.text.prepare(sentences)
        if not texts:
            return partials
        model_input = self.inputs.prepare(texts, model_positions, window_size=None)
        if not model_input:
            return partials
        predictions, _confidences = self.session.predict(model_input, texts)
        for sentence_id, position, prediction in zip(sentence_ids, result_positions, predictions):
            partials[sentence_id][position] = self.text.convert_bopomofo(prediction)
        return partials

    def close(self):
        self.session.close()
