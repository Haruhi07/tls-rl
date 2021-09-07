from argparse import ArgumentParser
from env_utils import extract_keywords
from environment import Environment
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from itertools import count
from utils import first_n_sents, format_decoder_input, show_gpu
from torch.distributions import Categorical
from dataset import build_dataloader
from collections import defaultdict

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


def main():
    parser = ArgumentParser()
    # Configuration
    parser.add_argument("--dataset", type=str, required=True)
    # RL
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--nfirst", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="sshleifer/distill-pegasus-cnn-16-4")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model_name = 'google/pegasus-multi_news'
    model_name = args.model_name
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    actor = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    state_size = tokenizer.vocab_size
    critic = Critic(state_size).to(device)
    critic_loss_fct = torch.nn.MSELoss()

    env = Environment(args, device)


    optimizerA = torch.optim.Adam(actor.lm_head.parameters())
    optimizerC = torch.optim.Adam(critic.parameters())

    def rl(cluster):
        input_ids = cluster['input_ids_dict']['input_ids'].to(device)
        source = cluster['source']
        for iter in range(args.episodes):
            rewards = []
            values = []
            returns = []
            actions = []
            batch = defaultdict(None)
            actor.eval()

            #input_ids = input_ids.to(device)
            # print(input_ids)

            # generate sample and calculate value
            with torch.no_grad():
                decoder_input_ids = [0]
                show_gpu("Before Sampling")
                while len(decoder_input_ids) < args.max_length:
                    decoder_input_ids_tensor = torch.LongTensor([decoder_input_ids]).to(device)
                    logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
                    #show_gpu("Everytime after generating next logits")
                    # print("logits = ", logits)
                    probs = F.softmax(logits, dim=-1)
                    action = torch.argmax(probs[0, -1], dim=-1)
                    actions.append(action)
                    # TODO: Add top_k here
                    # print("action = ", action)
                    decoder_input_ids = decoder_input_ids + [action.item()]
                    output = tokenizer.decode(decoder_input_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
                    batch['input_ids'] = input_ids
                    batch['decoder_input_ids'] = decoder_input_ids_tensor
                    batch['source'] = source
                    batch['summary'] = output
                    # calculate the reward of the sample
                    reward = env.calc_reward(batch)
                    rewards.append(reward)

                    if action == 1:
                        break

                logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
                last_state = logits[0, -1]
                # print("last_state = ", last_state)
                last_value = critic(last_state)

            print(output)
            print("iter = {} reward = {}".format(iter, reward))

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
            log_probs = [torch.reshape(d.log_prob(a), (-1, 1))[0] for d, a in zip(distributions, actions)]
            # print("distribution = ", distributions)
            # print("log_probs before cat = ", log_probs)

            # calculate values and returns
            ret = last_value
            for step in reversed(range(len(rewards))):
                ret = rewards[step] + args.gamma * ret
                returns.append(ret)
                values.append(critic(final_logits[0, step].detach()))

            # concatenate values, returns and log_probs
            # print("values before cat = ", values)
            # print("returns before cat = ", returns)
            log_probs = torch.cat(log_probs)
            # print("log_probs = ", log_probs)
            rewards = torch.FloatTensor(rewards).to(device)
            # print("rewards = ", rewards)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            # print("log_probs size = ", log_probs.size())
            # print("values size = ", values.size())
            # print("returns size = ", returns.size())

            advantages = rewards.detach() - values.detach()
            # print("advantages = ", advantages)

            critic_loss = critic_loss_fct(values, rewards)
            optimizerC.zero_grad()
            #critic_loss.backward(retain_graph=True)
            critic_loss.backward()

            # norm_rewards = (rewards.detach() - values.detach())
            # actor_loss = torch.mean(log_probs.mul(norm_rewards))
            actor_loss = -(log_probs * advantages.detach()).mean()

            print("actor_loss = ", actor_loss)
            print("critic_loss = ", critic_loss)

            optimizerA.zero_grad()
            actor_loss.backward()

            optimizerA.step()
            optimizerC.step()

            torch.cuda.empty_cache()
            del decoder_input_ids_tensor
            del distributions
            del decoder_input_ids
            del probs
            del output
            del last_state
            del last_value
            del log_probs
            del logits
            del final_logits
            del rewards
            del values
            del returns
            del actions
            del advantages

        print("final reward = ", reward)
        return reward

    data_loader = build_dataloader(args, tokenizer)
    for epoch in range(args.epochs):
        for data in data_loader:
            print(data)
            topic, clusters, timelines = data
            print("topic: ", topic)
            # Define Environment
            keywords = extract_keywords(timelines)
            env.update_keywords(keywords)
            print("env initialized...")
            show_gpu("Before training: ")
            for c in clusters.items():
                date, tokenized_cluster = c
                print(tokenized_cluster)
                reward = rl(tokenized_cluster)
                print(reward)

if __name__ == "__main__":
    main()