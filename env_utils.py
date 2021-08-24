import pathlib
import json
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(timeline, metric = 'tfidf'):
    text = [item["text"][0].lower() for item in timeline]
    if metric == "tfidf":
        vectorizer = TfidfVectorizer(stop_words = 'english')
        tfidf = vectorizer.fit_transform(text)
        vocab = vectorizer.get_feature_names() # vocabulary
        weight = tfidf.toarray() # weight[i][j] -- vocab[j]'s tf-idf value in text[i]
        keywords = []
        # take words with n-largest tfidf value
        for i in range(len(text)):
            idx = heapq.nlargest(10, range(len(weight[i])), weight[i].take)
            for id in idx:
                keywords.append(vocab[id])
        return set(keywords)

def concatenate(timeline):
    ret = ""
    for item in timeline:
        ret += item["text"]
    return ret


if __name__ == "__main__":
    print(extract_keywords("./dataset/t1/bpoil_bbc/bbc.txt.timeline.json"))