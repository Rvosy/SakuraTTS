"""Offline English G2P using prepared dictionaries, POS data and NumPy weights.

Adapted from GPT-SoVITS text/english.py (MIT, RVC-Boss) and g2p-en 2.1.0
(Apache-2.0, Kyubyong Park and Jongseok Kim). See docs/third-party/english.md.
Resource loading replaces their global initialization and automatic downloads.
"""

import json
from pathlib import Path
import re

import numpy as np


class EnglishG2P:
    def __init__(self, resources, symbols):
        from nltk.tag.perceptron import PerceptronTagger
        from nltk.tokenize import TweetTokenizer
        from wordsegment import Segmenter
        from ._vendor.english_normalization import normalize

        resources = Path(resources)
        data = json.loads((resources / "g2p.json").read_text(encoding="utf-8"))
        self.cmu = data["cmu"]
        self.namedict = data["namedict"]
        self.homograph2features = data["homographs"]
        self.g2idx = {g: i for i, g in enumerate(data["graphemes"])}
        self.idx2p = dict(enumerate(data["phonemes"]))
        with np.load(resources / "checkpoint.npz", allow_pickle=False) as weights:
            for name in ("enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh",
                         "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh", "fc_w", "fc_b"):
                setattr(self, name, weights[name])
        tagger = PerceptronTagger(load=False)
        tagger.model.weights = data["tagger"]["weights"]
        tagger.tagdict = data["tagger"]["tagdict"]
        tagger.classes = set(data["tagger"]["classes"])
        tagger.model.classes = tagger.classes
        self.tag = tagger.tag
        self.tokenize = TweetTokenizer().tokenize
        self.splitter = Segmenter()
        self.splitter.load()
        self.normalize_numbers = normalize
        self.symbols = set(symbols)

    def normalize(self, text):
        # Preserve the pinned upstream's literal punctuation replacements.
        replacements = {"[;:：，；]": ",", '["’]': "'", "。": ".", "！": "!", "？": "?"}
        pattern = re.compile("|".join(re.escape(p) for p in replacements))
        text = pattern.sub(lambda match: replacements[match.group()], text)
        text = self.normalize_numbers(text)
        marks = "".join(re.escape(p) for p in ("!", "?", "…", ",", ".", "-"))
        return re.sub(f"([{marks}\\s])([{marks}])+", r"\1", text)

    def g2p(self, text):
        phones = ["UNK" if p == "<unk>" else p for p in self(text)
                  if p not in (" ", "<pad>", "UW", "</s>", "<s>")]
        return [p if p in self.symbols else "-" for p in phones if p in self.symbols or p == "'"]

    def __call__(self, text):
        words = self.tokenize(text)
        tokens = self.tag(words)
        prons = []
        for o_word, pos in tokens:
            word = o_word.lower()
            if re.search('[a-z]', word) is None:
                pron = [word]
            elif len(word) == 1:
                if o_word == 'A':
                    pron = ['EY1']
                else:
                    pron = self.cmu[word][0]
            elif word in self.homograph2features:
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                elif len(pos) < len(pos1) and pos == pos1[:len(pos)]:
                    pron = pron1
                else:
                    pron = pron2
            else:
                pron = self.qryword(o_word)
            prons.extend(pron)
            prons.extend([' '])
        return prons[:-1]

    def qryword(self, o_word):
        word = o_word.lower()
        if len(word) > 1 and word in self.cmu:
            return self.cmu[word][0]
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]
        if len(word) <= 3:
            phones = []
            for w in word:
                if w == 'a':
                    phones.extend(['EY1'])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones
        if re.match("^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            if phones[-1] in ['P', 'T', 'K', 'F', 'TH', 'HH']:
                phones.extend(['S'])
            elif phones[-1] in ['S', 'Z', 'SH', 'ZH', 'CH', 'JH']:
                phones.extend(['AH0', 'Z'])
            else:
                phones.extend(['Z'])
            return phones
        comps = self.splitter.segment(word.lower())
        if len(comps) == 1:
            return self.predict(word)
        return [phone for comp in comps for phone in self.qryword(comp)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh
        rz_ih, n_ih = (rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:])
        rz_hh, n_hh = (rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:])
        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)
        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h
        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)
            outputs[:, t, :] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ['</s>']
        x = [self.g2idx.get(char, self.g2idx['<unk>']) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)
        return x

    def predict(self, word):
        enc = self.encode(word)
        enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh, self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
        last_hidden = enc[:, -1, :]
        dec = np.take(self.dec_emb, [2], axis=0)
        h = last_hidden
        preds = []
        for i in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3:
                break
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)
        preds = [self.idx2p.get(idx, '<unk>') for idx in preds]
        return preds
