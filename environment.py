import env_utils
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from collections import Counter


class Environment:
    def __init__(self, args, keywords):
        self.keywords = set(keywords)

    def topical_coherence(self, source, summary):
        pass

    def factual_consistency(self, summary):
        tokens = text.lower().split()
        ret = 0
        for t in tokens:
            if t in self.keywords:
                ret += 1
        return ret

    def punish_repetition(self, text):
        tokens = text.lower().split()
        cnt = Counter(tokens)
        rep = 0
        for k in cnt:
            v = cnt[k]
            if v > 1:
                rep += v
        return 1 - 1.0 * rep / len(tokens)

    def language_quality(self, input_ids, decoder_input_ids):
        pass

    def calc_reward(self, batch):
        source = batch['source']
        summary = batch['summary']
        input_ids = batch['input_ids']
        decoder_input_ids = batch['decoder_input_ids']
        ret = 0
        return ret
