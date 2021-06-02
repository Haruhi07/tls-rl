import collections
import datetime
import json
import os
import pathlib
import pickle

import numpy as np
from argparse import ArgumentParser

from sklearn.cluster import AffinityPropagation

from classes import Article, Cluster
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def load_articles(topic_path):
    articles_path = topic_path / "articles.json"
    with open(articles_path, "r") as articles_json:
        articles = json.load(articles_json)
    ret = []
    for article in articles:
        ret.append(Article(article))
    return ret


def calc_sim(X, metric='euclidean'):
    if metric == 'euclidean':
        return euclidean_distances(X)
    else:
        return cosine_distances(X)


def str2datetime(s):
    t = None
    formats = ['%Y-%m-%d', '%Y-%m', '%Y']
    for time_format in formats:
        try:
            t = datetime.datetime.strptime(date, time_format)
        except:
            pass
    return t


def main():
    parser = ArgumentParser()
    # Load arguments
    parser.add_argument("--dataset", type=str, default="./dataset/t17")
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    for topic in os.listdir(dataset_path):
        print("clustering topic: ", topic)

        topic_path = dataset_path / topic
        articles = load_articles(topic_path)
        n_articles = len(articles)
        article_embeddings = []
        for article in articles:
            sentences = article.text.split('.')
            sentences_embedding = embedding_model.encode(sentences)
            article_embedding = np.mean(sentences_embedding, axis=0)
            article_embeddings.append(article_embedding)

        embedding_matrix = np.vstack(article_embeddings)
        # calculating similarity (can be removed)
        #sim = -1 * calc_sim(embedding_matrix, metric='euclidean')
        #print(sim)
        af = AffinityPropagation(random_state = 0).fit(embedding_matrix)
        centers = af.cluster_centers_indices_
        labels = af.labels_
        n_centers = len(centers)

        clusters = collections.defaultdict(Cluster)
        for i in range(n_articles):
            clusters[centers[labels[i]]].articles.append(articles[i])

        # Assigning dates to clusters
        for c in clusters:
            cluster = clusters[c]
            date_count = collections.defaultdict(int)
            max_count = 0
            max_date = str2datetime(cluster.articles[0].dct)
            for article in cluster.articles:
                for date in article.dates:
                    t = str2datetime(date)
                    date_count[t] += 1
                    if date_count[t] > max_count:
                        max_count = date_count[t]
                        max_date = t
            cluster.date = max_date
            print(cluster.date)
        # Ranking all clusters

        # Saving ranked clusters
        cluster_path = topic_path / "cluster.pkl"
        with open(cluster_path, "wb") as f:
            pickle.dump(clusters, f)


if __name__=="__main__":
    main()