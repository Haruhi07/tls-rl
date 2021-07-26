import os
import math
import logging
import pathlib
import random
import time
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from ignite.engine import Engine, Events
from transformers import GPT2Tokenizer, GPT2Model, AdamW
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm.notebook import tnrange
from utils import concatenate, tokenize_dataset


logger = logging.getLogger()
MODEL_INPUTS = ["articles", "timelines"]

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make cudnn work in a fixed mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_dataset(tokenizer, dataset_path, cache_path=None):
    dataset_path = pathlib.Path(dataset_path)
    cache_path = pathlib.Path(cache_path) / type(tokenizer).__name__
    # Load cache dataset / Tokenize raw dataset and cache it
    if os.path.isfile(cache_path):
        logger.info("Loading tokenized dataset from cache at %s", cache_path)
        corpus = torch.load(cache_path)
    else:
        logger.info("Initializing raw dataset from %s", dataset_path)
        topics, clustered_articles, timelines = tokenize_dataset(tokenizer, dataset_path)
        corpus = []
        for topic, articles, timeline in zip(topics, clustered_articles, timelines):
            corpus.append({"topic": topic, "articles": articles, "timeline": timeline})
        print(corpus)
        torch.save(corpus, cache_path)
    return corpus

def get_dataloader(tokenizer, args):
    dataset_path = args.dataset_path
    cache_path = args.cache_path
    batch_size = args.batch_size
    corpus = initialize_dataset(tokenizer, dataset_path, cache_path)
    # build tensor datasets and DataLoader
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    valid_topics = ["bpoil_bbc"]
    for dataset in corpus:
        topic, articles, timelines = dataset
        dataset_class = "valid" if topic in valid_topics else "train"
        datasets[dataset_class]["topic"] = topic
        datasets[dataset_class]["articles"] = articles
        datasets[dataset_class]["timelines"] = timelines

    logger.info("Pad inputs and convert to Tensor")
    tensor_dataset = {"train": [], "valid": []}
    for dataset_class, dataset in datasets.items():
        print(dataset_class, dataset)
        #dataset = pad_dataset(dataset)
        for input_name in MODEL_INPUTS:
            print(input_name, dataset[input_name])
            tensor = torch.tensor(dataset[input_name])
            tensor_dataset[dataset_class].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_dataset["train"]), TensorDataset(*tensor_dataset["valid"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Train dataset {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset {}".format(valid_dataset.tensors[0].shape))


def train():
    parser = ArgumentParser()
    # Arguments of the Model
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")

    # Generation
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    # RL - Critic


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))
    logger.info("Initialize tokenizer, pretrained model and optimizer")

    reset_seed(args.seed)

    # basis_model_name = 'google/pegasus-multi_news'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2Model.from_pretrained("distilgpt2").to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    logger.info("Initialize datasets")
    get_dataloader(tokenizer, args)

    # Training function and trainer
    def rl_update(engine, batch):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        model.eval()

        # Get all information in the current batch for reconstruction
        batch_dict = {}
        batch_dict['topic'] = batch[0].cpu().detach().numpy()
        batch_dict['clusters']


if __name__ == "__main__":
    train()