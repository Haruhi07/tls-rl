import json
import os
import pickle

from torch.utils.data import Dataset


# 一个cluster是一次训练的一整个输入
class ClusteredDataset(Dataset):
    def __init__(self, dataset_path):
        self.topics = []
        self.clusters = []
        self.timelines = []  # element of this should also be list to solve multiple timelines
        for topic in os.listdir(dataset_path):
            self.topics.append(topic)
            cluster_path = dataset_path / topic / "cluster.pkl"
            with open(cluster_path, "rb") as f:
                cluster_list = pickle.load(f)
            self.clusters.appen(cluster_list)

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
        return self.clusters[idx], self.timelines[idx]
