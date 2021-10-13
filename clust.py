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
        ret = euclidean_distances(X)
    else:
        # 1 - cos_sim(X)
        ret = cosine_distances(X)
    return ret


def str2datetime(s):
    t = None
    formats = ['%Y-%m-%d', '%Y-%m', '%Y']
    for time_format in formats:
        try:
            t = datetime.datetime.strptime(s, time_format)
        except:
            pass
    return t


def main():
    parser = ArgumentParser()
    # Load arguments
    parser.add_argument("--dataset", type=str, default="./dataset/t1")
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    for topic in os.listdir(dataset_path):
        print("clustering topic: ", topic)
        topic_path = dataset_path / topic
        if not topic_path.is_dir() or topic == '.DS_Store':
            continue

        articles = load_articles(topic_path)
        n_articles = len(articles)
        article_embeddings = []

        for article in articles:
            dates = [d for d in article.dates if d != None]
            if len(dates) == 0:
                #print('dct added!')
                dates = [article.dct]
            article.dates = set(dates)
            #print(f'article.dates={article.dates}')
            sentences = article.text.split('.')
            sentences_embedding = embedding_model.encode(sentences)
            article_embedding = np.mean(sentences_embedding, axis=0)
            article_embeddings.append(article_embedding)

        embedding_matrix = np.vstack(article_embeddings)

        sim = -1 * calc_sim(embedding_matrix, metric='euclidean')
        for i, ai in enumerate(articles):
            for j, aj in enumerate(articles):
                compatible = False
                for t in ai.dates:
                    if t in aj.dates:
                        compatible = True
                        break
                if not compatible:
                    sim[i][j] = -100000

        af = AffinityPropagation(preference=-50, affinity='precomputed', random_state=None).fit(sim)
        centers = af.cluster_centers_indices_
        labels = af.labels_
        n_centers = len(centers)
        print(n_centers)

        clusters = collections.defaultdict(Cluster)
        for i in centers:
            clusters[i].centroid = articles[i]
        for i in range(n_articles):
            clusters[centers[labels[i]]].articles.append(articles[i])

        # Assigning dates to clusters
        clusters_list = []
        for c in clusters:
            cluster = clusters[c]
            print(f'c={c} cluster={cluster}')
            date_count = collections.defaultdict(int)
            max_count = 0
            max_date = None
            for article in cluster.articles:
                print(article.dates)
                for d in article.dates:
                    t = str2datetime(d)
                    date_count[t] += 1
                    if date_count[t] > max_count:
                        max_count = date_count[t]
                        max_date = t
            print(f'max_date={max_date}')
            cluster.date_count = max_count
            cluster.date = max_date
            clusters_list.append(cluster)

        # Ranking all clusters
        def get_count_date(cluster):
            return cluster.date_count
        clusters_list = sorted(clusters_list, reverse=True, key=get_count_date)
        for cluster in clusters_list:
            print(cluster.articles, cluster.date, cluster.date_count)

        # Saving ranked clusters
        cluster_path = topic_path / "cluster.pkl"
        with open(cluster_path, "wb") as f:
            pickle.dump(clusters_list, f)


if __name__=="__main__":
    main()