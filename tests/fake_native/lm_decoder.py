"""Pure-python stand-in for the native lm_decoder extension; tests only, never a real decoder.

Words are "w<class>" for each new non-blank, non-silence WFST-order argmax (CTC-style collapse).
"""
import numpy as np


class DecodeOptions:
    def __init__(self, max_active, min_active, beam, lattice_beam, acoustic_scale,
                 blank_skip_thresh, length_penalty, nbest):
        self.acoustic_scale = acoustic_scale


class DecodeResource:
    def __init__(self, tlg_path, g_path, rescore_path, words_path, unit_path):
        self.paths = (tlg_path, words_path)


class DecodeResult:
    def __init__(self, sentence):
        self.sentence, self.ac_score, self.lm_score = sentence, 0.0, 0.0


class BrainSpeechDecoder:
    def __init__(self, resource, options):
        self.Reset()

    def Reset(self):
        self.tokens, self.previous, self.finished = [], None, False

    def FinishDecoding(self):
        self.finished = True

    def DecodedSomething(self):
        return bool(self.tokens)

    def result(self):
        return [DecodeResult(" ".join(f"w{t}" for t in self.tokens))] if self.tokens else []


def DecodeNumpy(decoder, logits, log_priors, blank_penalty):
    rows = np.asarray(logits, dtype=np.float32).copy()
    rows[:, 0] -= blank_penalty
    for row in rows:
        best = int(np.argmax(row))
        if best != decoder.previous and best not in (0, 1):
            decoder.tokens.append(best)
        decoder.previous = best
