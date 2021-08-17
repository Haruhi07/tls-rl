import json
import os
import pickle
import pathlib

from torch.utils.data import Dataset, DataLoader


# 一个cluster是一次训练的一整个输入
class ClusteredDataset(Dataset):
    def __init__(self, dataset_path, tokenizer):
        self.tokenizer = tokenizer
        self.topics = []
        self.clusters = []
        self.timelines = []  # element of this should also be list to solve multiple timelines
        for topic in os.listdir(dataset_path):
            topic_path = dataset_path / topic
            if not topic_path.is_dir():
                continue
            print("adding {} into dataset...".format(topic))
            self.topics.append(topic)
            cluster_path = dataset_path / topic / "cluster.pkl"
            with open(cluster_path, "rb") as f:
                cluster_list = pickle.load(f)
            self.clusters.append(cluster_list)

            for file in os.listdir(dataset_path / topic):
                if 'timeline' not in file:
                    continue
                timeline_path = dataset_path / topic / file
                with open(timeline_path, "r") as f:
                    timeline = json.load(f)
                self.timelines.append(timeline)

    def __len__(self):
        return len(self.topics)

    def __getitem__(self, idx):
        cluster = {}
        for c in self.clusters[idx]:
            date = c.date
            articles = [a.text for a in c.articles]
            cluster[date] = self.tokenizer(articles, truncation=True, padding='longest', return_tensors='pt')
        return self.topics[idx], cluster, self.timelines[idx]

def build_dataloader(args, tokenizer):
    dataset_path = pathlib.Path(args.dataset)
    dataset = ClusteredDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    for data in dataloader:
        print(data)