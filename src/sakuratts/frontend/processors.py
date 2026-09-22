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
