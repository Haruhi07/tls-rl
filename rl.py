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


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size

        self.linear1 = nn.Linear(self.state_size, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        value = self.linear2(output)
        return value

def setup_env(tokenizer, args):
    dataset_path = pathlib.Path(args.dataset)
    keywords = []
    length = 0
    with open(dataset_path / "cluster.pkl", 'rb') as f:
        clusters = pickle.load(f)

    for file in os.listdir(dataset_path):
        if "timeline" in file:
            keywords = extract_keywords(dataset_path / file)
            with open(dataset_path /file, 'rb') as f:
                timelines = json.load(f)
            length = len(timelines)
            print("length = ", length)
    env = Environment(tokenizer, clusters, keywords, length)
    return env

def compute_returns(next_value, rewards, masks, gamma):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def get_logits(observation, nfirst):
    cluster, timeline = observation
    print(timeline)

    encoder_input = [first_n_sents(a.text, nfirst) for a in cluster.articles]
    decoder_input = timeline["text"]

    encoder_input_ids = tokenizer(encoder_input, padding=True, truncation=True, return_tensors="pt").input_ids.to(dvc)
    decoder_input_ids = format_decoder_input(tokenizer(decoder_input, return_tensors="pt").input_ids).to(dvc)
    print(decoder_input_ids)

    logits = actor.forward(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids).logits  # state
    return logits

def generate(input_ids, actor, device, args):

    print("probs = ", probs)
    return probs, decoder_input_ids

def main():
    parser = ArgumentParser()
    # Configuration
    parser.add_argument("--dataset", type=str, required=True)
    # RL
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--nfirst", type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_ids = range(torch.cuda.device_count())
    model_name = 'sshleifer/distill-pegasus-cnn-16-4'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    actor = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    state_size = tokenizer.vocab_size
    critic = Critic(state_size).to(device)
    critic_loss_fct = torch.nn.MSELoss()

    # Define Environment
    env = setup_env(tokenizer, args)
    print("env initialized...")

    optimizerA = torch.optim.Adam(actor.lm_head.parameters())
    optimizerC = torch.optim.Adam(critic.parameters())

    for iter in range(args.episodes):
        rewards = []
        values = []
        returns = []
        actions = []
        env.reset()
        actor.eval()
        cluster, timeline = env.observation()
        # sample
        input = ' '.join([first_n_sents(a.text, args.nfirst) for a in cluster.articles]) + '.'
        #print(input)
        input_ids = tokenizer(input, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        #print(input_ids)

        # generate sample and calculate value
        with torch.no_grad():
            decoder_input_ids = [0]
            while len(decoder_input_ids) < args.max_length:
                decoder_input_ids_tensor = torch.LongTensor([decoder_input_ids]).to(device)
                logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
                print("logits = ", logits)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs[0, -1], dim=-1)
                actions.append(action)
                # TODO: Add top_k here
                print("action = ", action)
                decoder_input_ids = decoder_input_ids + [action.item()]
                output = tokenizer.decode(decoder_input_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
                # calculate the reward of the sample
                reward = env.count_keyword(output)
                rewards.append(reward)
                print("reward = ", reward)

                if action == 1:
                    break
            print(output)

            logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
            last_state = logits[0, -1]
            print("last_state = ", last_state)
            last_value = critic(last_state)

        # only tune the lm_head layer
        actor.eval()
        actor.lm_head.train()
        for p in actor.parameters():
            p.requires_grad = False
        for p in actor.lm_head.parameters():
            p.requires_grad = True

        # create calculation graph with gradient on lm_head
        final_logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
        print("final_logits = ", final_logits)
        distributions = [Categorical(F.softmax(lgt, dim=-1)) for lgt in final_logits[0]]
        log_probs = [torch.reshape(d.log_prob(a), (-1,1))[0] for d, a in zip(distributions, actions)]
        print("distribution = ", distributions)
        print("log_probs before cat = ", log_probs)

        # calculate values and returns
        ret = last_value
        for step in reversed(range(len(rewards))):
            ret = rewards[step] + args.gamma * ret
            returns.append(ret)
            values.append(critic(final_logits[0, step]))

        # concatenate values, returns and log_probs
        #print("values before cat = ", values)
        #print("returns before cat = ", returns)
        log_probs = torch.cat(log_probs)
        print("log_probs = ", log_probs)
        rewards = torch.FloatTensor(rewards).to(device)
        print("rewards = ", rewards)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        print("log_probs size = ", log_probs.size())
        print("values size = ", values.size())
        print("returns size = ", returns.size())

        advantages = returns.detach() - values.detach()
        print("advantages = ", advantages)

        critic_loss = critic_loss_fct(values, rewards)
        optimizerC.zero_grad()
        critic_loss.backward(retain_graph=True)
        optimizerC.step()

        #norm_rewards = (rewards.detach() - values.detach())
        #actor_loss = torch.mean(log_probs.mul(norm_rewards))
        actor_loss = -(log_probs * advantage.detach()).mean()

        print("actor_loss = ", actor_loss)
        print("critic_loss = ", critic_loss)

        optimizerA.zero_grad()
        actor_loss.backward()
        optimizerA.step()

    print("final reward = ", reward)

if __name__ == "__main__":
    main()