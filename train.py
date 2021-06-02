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
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

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
def sl_train():
    train_sampler = RandomSampler()

def rl_update(engine, batch):
    pass

rl_trainer = Engine(rl_update)

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
    epoch_times = sl_train(model, tokenizer, dataset)
    end_time = time.time()
    print('time: ', (end_time - start_time) / 60, ' minutes', end='\n\n')

    print('Saving trained model...\n')
    model_folder = pathlib.Path(args.model_dir)
    if not model_folder.exists():
        os.mkdir(model_folder)
    model_file = model_folder / "model_{}_data_{}_trained_after_{}_epochs.pkl".format(model_name, os.basename(args.dataset), epoch_times)
    config_file = model_folder / "config_{}_data_{}_trained_after_{}_epochs.json".format(model_name, os.basename(args.dataset), epoch_times)
    torch.save(model.state_dict(), model_file)
    model.config.to_json_file(config_file)


if __name__ == "__main__":
    main()
