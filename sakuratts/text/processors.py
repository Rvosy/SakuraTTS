"""Language-specific phones and features consumed by the shared frontend."""

import numpy as np


class JapaneseProcessor:
    def __init__(self, g2p):
        self.g2p = g2p

    def clean(self, text):
        normalized = self.g2p.normalize(text)
        return self.g2p.g2p(normalized), None, normalized

    def features(self, phones, word2ph, normalized):
        # The official Japanese input has no BERT model dependency.
        return np.zeros((1024, len(phones)), dtype=np.float32)


class EnglishProcessor(JapaneseProcessor):
    """English also uses zero BERT features in the upstream V2 frontend."""


class ChineseProcessor:
    """Load Mandarin resources when the first Chinese segment reaches the frontend."""

    def __init__(self, directory):
        self.directory = directory
        self.phones = None
        self.bert = None

    def clean(self, text):
        if self.phones is None:
            from .chinese import ChinesePhones
            from .g2pw import G2PW
            g2pw = G2PW(self.directory, self.directory / "tokenizer.json", self.directory / "g2pw.onnx")
            try:
                self.phones = ChinesePhones(self.directory, g2pw)
            except BaseException:
                g2pw.close()
                raise
        return self.phones.clean(text)

    def features(self, phones, word2ph, normalized):
        from .chinese import chinese_bert_features
        if self.bert is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.bert = ort.InferenceSession(str(self.directory / "bert.onnx"), sess_options=options,
                                             providers=["CPUExecutionProvider"])
        return chinese_bert_features(normalized, word2ph, self._hidden, self.phones.g2pw.tokenizer)

    def _hidden(self, **inputs):
        return self.bert.run(["hidden"], inputs)[0]

    def close(self):
        self.bert = None
        if self.phones is not None:
            self.phones.g2pw.close()
            self.phones = None
