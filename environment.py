import env_utils
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter


class Environment:
    def __init__(self, args, device, keywords=None):
        self.keywords = None
        self.keywords_embeddings = None
        self.args = args
        self.lq_evaluater = PegasusForConditionalGeneration.from_pretrained(args.model_name).to(device)
        self.encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.weights = [0.25, 0.25, 0.25, 0.25]

    def update_keywords(self, keywords):
        self.keywords = list(keywords)
        self.keywords_embeddings = self.encoder.encode(self.keywords)

    # R1
    def factual_consistency(self, source, summary):
        source_embedding = self.encoder.encode(source)
        summary_embedding = self.encoder.encode([summary])
        print("source_embedding = ", source_embedding)
        print("summary_embedding = ", summary_embedding)
        ret = cosine_similarity(source_embedding, summary_embedding)
        print("R1 = ", ret)
        return ret

    # R2
    def topical_coherence(self, summary_embedding):
        ret = cosine_similarity(self.keywords_embeddings, summary_embedding)
        print("R2 = ", ret)
        return ret

    # R3
    def language_quality(self, input_ids, decoder_input_ids):
        loss = self.lq_evaluater(input_ids=input_ids, labels=decoder_input_ids).loss
        loss = (self.args.alpha - loss) / self.args.alpha
        return loss

    # R4
    def repetition_punishment(self, summary):
        tokens = summary.lower().split()
        cnt = Counter(tokens)
        rep = 0
        for k in cnt:
            v = cnt[k]
            if v > 1:
                rep += v
        return 1 - 1.0 * rep / len(tokens)

    def calc_reward(self, batch):
        source = batch['source']
        summary = batch['summary']
        input_ids = batch['input_ids']
        decoder_input_ids = batch['decoder_input_ids']

        summary_embedding = self.encoder.encode([summary])

        ret = self.weights[0] * self.topical_coherence(summary_embedding=summary_embedding) \
            + self.weights[1] * self.factual_consistency(source=source, summary=summary) \
            + self.weights[2] * self.language_quality(input_ids=input_ids, decoder_input_ids=decoder_input_ids) \
            + self.weights[3] * self.repetition_punishment(summary=summary)
        return ret
