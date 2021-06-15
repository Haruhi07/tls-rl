import os
import math
import logging
import pathlib
import random
import time
from pprint import pformat
from argparse import ArgumentParser

import numpy as np
import torch
from ignite.engine import Engine, Events
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm.notebook import tnrange
from utils import concatenate

from dataset import ClusteredDataset

logger = logging.getLogger(__file__)


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


# Training function and trainer
def sl_train(device, model, tokenizer, dataset, num_workers, lr, n_epochs, seed):
    train_sampler = RandomSampler(dataset)
    train_dl = DataLoader(dataset, sampler=train_sampler, num_workers=num_workers)
    loss_fct = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=-1)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = tnrange(n_epochs, desc="Epoch")
    reset_seed(seed)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            clustered_articles, ref_timelines = batch
            n_timelines = len(ref_timelines)
            print("n_timelines: ", n_timelines)
            tgt_timeline = []
            used_date = []
            for article in clustered_articles:
                inputs, date = article
                print(date)
                model.train()
                batch = tokenizer(inputs, truncation=True, padding='longest', return_tensors="pt").to(device)
                translated = model.generate(**batch)
                summary = tokenizer.batch_decode(translated, skip_special_tokens=True)#
                date = date[0].split()[0]
                if date not in used_date:
                    tgt_timeline.append({'date': date, 'text': summary[0]})
                if len(tgt_timeline) == n_timelines:
                    break

            print(tgt_timeline)
            cct_tgt_timeline = concatenate(tgt_timeline)
            cct_ref_timeline = concatenate(ref_timelines)
            loss = loss_fct(cct_ref_timeline, cct_tgt_timeline)
            loss.backward()
            # help to prevent gradient explosion or vanish
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            logging_loss = tr_loss
            print("loss: ", loss.item(), end='\n\n')



def rl_update(engine, batch):
    pass


# Evaluation function and evaluator
def inference(engine, batch):
    pass

    evaluator = Engine(inference)

    # Attach evaluation to rl_trainer

    # Decrease the learning rate


def main():
    parser = ArgumentParser()
    # Configuration
    parser.add_argument("--model_dir", type=str, default="./models/")
    parser.add_argument("--dataset", type=str, default="./dataset/t1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=1)
    # PEGASUS
    # Generation
    parser.add_argument("--seed", type=int, default=7)
    # RL - Critic
    args = parser.parse_args()

    # Set logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    # Do as the info below
    logger.info("Initialize tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    reset_seed(args.seed)

    logger.info("Initialize datasets")

    dataset_path = pathlib.Path(args.dataset)
    dataset = ClusteredDataset(dataset_path)
    model_name = 'google/pegasus-multi_news'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    start_time = time.time()
    sl_train(device, model, tokenizer, dataset, args.num_workers, args.lr, args.n_epochs, args.seed)
    end_time = time.time()
    print('time: ', (end_time - start_time) / 60, ' minutes', end='\n\n')

    print('Saving trained model...\n')
    model_folder = pathlib.Path(args.model_dir)
    if not model_folder.exists():
        os.mkdir(model_folder)
    model_file = model_folder / "model_{}_data_{}_trained_after_{}_epochs.pkl".format(model_name, os.path.basename(args.dataset), arg.n_epochs)
    config_file = model_folder / "config_{}_data_{}_trained_after_{}_epochs.json".format(model_name, os.path.basename(args.dataset), arg.n_epochs)
    torch.save(model.state_dict(), model_file)
    model.config.to_json_file(config_file)


if __name__ == "__main__":
    main()
