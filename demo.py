from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from datasets import load_dataset


dataset_name = 'cnn_dailymail'
dataset_version = '3.0.0'
model_name = 'sshleifer/distill-pegasus-cnn-16-4'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset(dataset_name, dataset_version, split='train')
print(dataset)
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

dataset = dataset.map(lambda e: tokenizer(e['article'], truncation=True, padding='max_length'), batched=True)
dataset = dataset.map(lambda e: tokenizer(e['highlights'], truncation=True, padding='max_length'), batched=True)
dataset.set_format(type='torch', columns=['article', 'highlights', 'id'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for it in dataloader:
    print(it)
    break