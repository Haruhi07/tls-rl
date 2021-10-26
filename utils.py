import os
import json
import pickle
import torch
import subprocess
import numpy as np
import gzip
import io
import datetime
import codecs
import tarfile
import pandas
import shutil
import collections
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def force_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def dict_to_dense_vector(d, key_to_idx):
    x = np.zeros(len(key_to_idx))
    for key, i in key_to_idx.items():
        x[i] = d[key]
    return x


def read_file(path):
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return text


def write_file(s, path):
    with open(path, 'w') as f:
        f.write(s)


def read_json(path):
    text = read_file(path)
    return json.loads(text)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(items, path, batch_size=100, override=True):
    if override:
        with open(path, 'w'):
            pass

    batch = []
    for i, x in enumerate(items):
        if i > 0 and i % batch_size == 0:
            with open(path, 'a') as f:
                output = '\n'.join(batch) + '\n'
                f.write(output)
            batch = []
        raw = json.dumps(x)
        batch.append(raw)

    if batch:
        with open(path, 'a') as f:
            output = '\n'.join(batch) + '\n'
            f.write(output)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj,  f)


def write_gzip(text, path):
    with gzip.open(path, 'wb') as output:
        with io.TextIOWrapper(output, encoding='utf-8') as enc:
            enc.write(text)


def read_gzip(path):
    with gzip.open(path, 'rb') as input_file:
        with io.TextIOWrapper(input_file) as dec:
            content = dec.read()
    return content


def read_jsonl_gz(path):
    with gzip.open(path, 'rb') as input_file:
        with io.TextIOWrapper(input_file) as dec:
            for line in dec:
                yield json.loads(line)

def read_tar_gz(path):
    contents = []
    with tarfile.open(path, 'r:gz') as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            content = f.read()
            contents.append(content)
    return contents


def read_json_tar_gz(path):
    contents = read_tar_gz(path)
    raw_data = contents[0]
    return json.loads(raw_data, strict=False)


def get_date_range(start, end):
    diff = end - start
    date_range = []
    for n in range(diff.days + 1):
        t = start + datetime.timedelta(days=n)
        date_range.append(t)
    return date_range


def days_between(t1, t2):
    return abs((t1 - t2).days)


def any_in(items, target_list):
    return any([item in target_list for item in items])


def csr_item_generator(M):
    """Generates tuples (i,j,x) of sparse matrix."""
    for row in range(len(M.indptr) - 1):
        i,j = M.indptr[row], M.indptr[row + 1]
        for k in range(i,j):
            yield (row, M.indices[k], M.data[k])


def max_normalize_matrix(A):
    try:
        max_ = max(A.data)
        for i, j, x in csr_item_generator(A):
            A[i, j] = x / max_
    except:
        pass
    return A


def gzip_file(inpath, outpath, delete_old=False):
    with open(inpath, 'rb') as infile:
        with gzip.open(outpath, 'wb') as outfile:
            outfile.writelines(infile)
    if delete_old:
        os.remove(inpath)


def normalise(X, method='standard'):
    if method == 'max':
        return X / X.max(0)
    elif method == 'minmax':
        return MinMaxScaler().fit_transform(X)
    elif method == 'standard':
        return StandardScaler().fit_transform(X)
    elif method == 'robust':
        return RobustScaler().fit_transform(X)
    else:
        raise ValueError('normalisation method not known: {}'.format(method))


def normalize_vectors(vector_batches, mode='standard'):
    if mode == 'max':
        normalize = lambda X: X / X.max(0)
    elif mode == 'minmax':
        normalize = lambda X: MinMaxScaler().fit_transform(X)
    elif mode == 'standard':
        normalize = lambda X: StandardScaler().fit_transform(X)
    elif mode == 'robust':
        normalize = lambda X: RobustScaler().fit_transform(X)
    else:
        normalize = lambda X: X
    norm_vectors = []
    for vectors in vector_batches:
        X = np.array(vectors)
        X_norm = normalize(X)
        norm_vectors += list(X_norm)
    return norm_vectors


def strip_to_date(t):
    return datetime.datetime(t.year, t.month, t.day)


def print_tl(tl):
    for t, sents in tl.items:
        print('[{}]'.format(t.date()))
        for s in sents:
            print(' '.join(s.split()))
        print('---')

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