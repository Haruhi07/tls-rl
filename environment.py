import env_utils
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter


class Environment:
    def __init__(self, args, device, keywords=None):
        self.keywords = None
        self.lq_evaluater = PegasusForConditionalGeneration.from_pretrained(args.model_name).to(device)
        self.encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.weights = [0.25, 0.25, 0.25, 0.25]

    def update_keywords(self, keywords):
        self.keywords = keywords

    def factual_consistency(self, source, summary):
        source_embedding = self.encoder.encode(source)
        summary_embedding = self.encoder.encode([summary])
        print("source_embedding = ", source_embedding)
        print("summary_embedding = ", summary_embedding)
        ret = cosine_similarity(source_embedding, summary_embedding)
        print("cos_sim = ", ret)
        return ret

    def topical_coherence(self, summary):
        tokens = summary.lower().split()
        ret = 0
        for t in tokens:
            if t in self.keywords:
                ret += 1
        return ret

    def repetition_punishment(self, summary):
        tokens = summary.lower().split()
        cnt = Counter(tokens)
        rep = 0
        for k in cnt:
            v = cnt[k]
            if v > 1:
                rep += v
        return 1 - 1.0 * rep / len(tokens)

    def language_quality(self, input_ids, decoder_input_ids):
        loss = self.lq_evaluater(input_ids=input_ids, labels=decoder_input_ids).loss
        loss = (args.alpha - loss) / args.alpha
        return loss

    def calc_reward(self, batch):
        source = batch['source']
        summary = batch['summary']
        input_ids = batch['input_ids']
        decoder_input_ids = batch['decoder_input_ids']

        ret = self.weights[0] * self.topical_coherence(summary=summary) \
            + self.weights[1] * self.factual_consistency(source=source, summary=summary) \
            + self.weights[2] * self.language_quality(input_ids=input_ids, decoder_input_ids=decoder_input_ids) \
            + self.weights[3] * self.repetition_punishment(summary=summary)
        return ret
