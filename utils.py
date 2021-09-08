import os
import json
import pickle
import torch
import subprocess


def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')

def first_n_sents(text, n=5):
    sentences = text.split('.')
    n_sentences = len(sentences)
    ret = '.'.join(sentences[:min(n, n_sentences)])
    print("ret == ", ret)
    return ret

def format_decoder_input(t):
    shape = list(t.size())
    shape[-1] = 1
    starts = torch.zeros(shape, dtype=torch.int)
    t = torch.cat((starts, t), -1)

    shape = list(t.size())
    last_dim = shape[-1]
    t = torch.split(t, [last_dim - 1, 1], -1)[0]
    return t

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