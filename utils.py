import os
import json
import pickle


def datetime2str(date):
    return str(date).split()[0]

def concatenate(timeline):
    cct_timeline = ''

    for summary in timeline:
        cct_timeline = cct_timeline + ' ' + summary['text'][0]

    return cct_timeline

def tokenize_dataset(tokenizer, dataset_path):
    topics = os.listdir(dataset_path)
    clustered_articles = []
    timelines = []
    for topic in topics:
        with open(dataset_path / topic / "cluster.pkl", "rb") as f:
            clusters = pickle.load(f)
            tokenized_clusters = []
            for c in clusters:
                date = c.date
                articles = [tokenizer(a.text, truncation=True)["input_ids"] for a in c.articles]
                tokenized_clusters.append({"date": datetime2str(date), "text": articles})
            clustered_articles.append(tokenized_clusters)

        for file in os.listdir(dataset_path / topic):
            if 'timeline' not in file:
                continue
            with open(dataset_path / topic / file, 'r') as f:
                timeline = json.load(f)
                cur_timeline = []
                for d in timeline:
                    date = d["date"]
                    summary = tokenizer(d["text"], truncation=True)["input_ids"]
                    cur_timeline.append({"date": date, "text": summary})
                timelines.append(cur_timeline)
    return topics, clustered_articles, timelines