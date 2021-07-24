from argparse import ArgumentParser
from env_utils import extract_keywords
from environment import Environment
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from itertools import count
from utils import first_n_sents, format_decoder_input
from torch.distributions import Categorical

import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle


def setup_env(tokenizer, args):
    dataset_path = pathlib.Path(args.dataset)
    keywords = []
    length = 0
    with open(dataset_path / "cluster.pkl", 'rb') as f:
        clusters = pickle.load(f)

    for file in os.listdir(dataset_path):
        if "timeline" in file:
            keywords.extend(extract_keywords(dataset_path / file))
            with open(dataset_path /file, 'rb') as f:
                timelines = json.load(f)
            length = len(timelines)
            print("length = ", length)
    env = Environment(tokenizer, clusters, keywords, length)
    return env

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size

        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def compute_returns(next_value, rewards, masks, gamma):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def get_logits(observation, tokenizer, actor, device, nfirst):
    cluster, timeline = observation

    encoder_input = [first_n_sents(a.text, nfirst) for a in cluster.articles]
    decoder_input = timeline["text"]

    encoder_input_ids = tokenizer(encoder_input, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    decoder_input_ids = format_decoder_input(tokenizer(decoder_input, return_tensors="pt").input_ids).to(device)

    logits = actor(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids).logits  # state
    return logits

def main():
    parser = ArgumentParser()
    # Configuration
    parser.add_argument("--dataset", type=str, required=True)
    # RL
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--nfirst", type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'google/pegasus-multi_news'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    actor = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    critic = Critic(tokenizer.vocab_size).to(device)
    
    # Define Environment
    env = setup_env(tokenizer, args)
    print("env initialized...")
    
    optimizerA = torch.optim.Adam(actor.parameters())
    optimizerC = torch.optim.Adam(critic.parameters())

    for iter in range(args.episodes):
        env.reset()
        observation = env.observation()
        log_probs = []
        values = []
        rewards = []
        masks = []
        
        for i in count():
            logits = get_logits(observation, tokenizer, actor, device, args.nfirst)

            dist = Categorical(F.softmax(logits, dim=-1))
            value = critic(logits)

            action = dist.sample()
            next_observation, reward, done = env.step(action)

            log_prob = dist.log_prob(action).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            observation = next_observation

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break

        logits = get_logits(next_observation, tokenizer, actor, device, args.nfirst)
        next_value = critic(logits)
        returns = compute_returns(next_value, rewards, masks, args.gamma)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()


if __name__ == "__main__":
    main()