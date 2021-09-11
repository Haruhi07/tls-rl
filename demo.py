from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from datasets import load_dataset
import pathlib


dataset_name = 'cnn_dailymail'
dataset_version = '3.0.0'
model_name = 'sshleifer/distill-pegasus-cnn-16-4'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cache_dir = pathlib.Path('/work/hs20307/huggingface')

dataset = load_dataset(dataset_name, dataset_version, split='test', cache_dir=cache_dir/'datasets')
print(dataset)
tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir=cache_dir/'transformers')
model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir/'transformers').to(device)

dataset = dataset.map(lambda e: tokenizer(e['highlights'], truncation=True, padding='max_length'), batched=True)
dataset = dataset.map(lambda e: {'labels': e['input_ids']})
dataset = dataset.map(lambda e: tokenizer(e['article'], truncation=True, padding='max_length'), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

max_loss = 0

for it in dataloader:
    result_dict = model(**it)
    loss_list = result_dict.loss
    batch_maxloss = max(loss_list)
    max_loss = max(max_loss, batch_maxloss)

print("max_loss = ", max_loss)