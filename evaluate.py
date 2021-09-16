from argparse import ArgumentParser
from pathlib import Path
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
from pprint import pprint
from dataset import build_dataloader
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from collections import defaultdict

import torch


train_topic = ['EgyptianProtest_cnn', 'haiti_bbc']


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./models')
    parser.add_argument("--result_path", type=str, default='./results')
    parser.add_argument("--model_name", type=str, default="sshleifer/distill-pegasus-cnn-16-4")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = pathlib.Path(args.model_path)
    result_path = pathlib.Path(args.result_path)
    cache_dir = pathlib.Path('/work/hs20307/huggingface')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path/'actor.pt').to(device)
    model.eval()

    result = defaultdict(list)

    data_loader = build_dataloader(args, tokenizer)
    for data in data_loader:
        topic, clusters, timelines = data
        print("topic: ", topic)
        if topic in train_topic:
            continue
        # Define Environment
        keywords = extract_keywords(timelines)
        env.update_keywords(keywords)
        print("env initialized...")
        show_gpu("Before training: ")
        for c in clusters.items():
            date, tokenized_cluster = c
            batch = tokenized_cluster['input_ids_dict']
            print(batch)
            output_ids = model.generate(**batch)
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(output_text)
            result[topic].append({date: output_text})



if __name__=="__main__":
    main()