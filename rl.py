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

    # Define Environment
    env = setup_env(tokenizer, args)
    print("env initialized...")

    optimizerA = torch.optim.Adam(actor.parameters())
    optimizerC = torch.optim.Adam(critic.parameters())

    rewards = []
    values = []
    returns = []
    for iter in range(args.episodes):
        env.reset()
        actor.eval()
        cluster, timeline = env.observation()
        # sample
        input = ' '.join([first_n_sents(a.text, args.nfirst) for a in cluster.articles]) + '.'
        print(input)
        input_ids = tokenizer(input, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        print(input_ids)

        # generate sample and calculate value
        rewards = []
        with torch.no_grad():
            decoder_input_ids = [0]
            while len(decoder_input_ids) < args.max_length:
                decoder_input_ids_tensor = torch.LongTensor([decoder_input_ids]).to(device)
                logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
                print("logits = ", logits)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs[0, -1], dim=-1).item()
                # TODO: Add top_k here
                print(action)
                decoder_input_ids = decoder_input_ids + [action]
                output = tokenizer.decode(decoder_input_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
                # calculate the reward of the sample
                reward = env.count_keyword(output)
                rewards.append(reward)
                print("reward = ", reward)

                if action == 1:
                    break

            logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids_tensor).logits
            last_state = logits[0, -1]
            print("last_state = ", last_state)
            last_value = critic(last_state)
            print("last_value = ", last_value)

        # only tune the lm_head layer
        actor.eval()
        actor.lm_head.train()
        for p in actor.parameters():
            p.requires_grad = False
        for p in actor.lm_head.parameters():
            p.requires_grad = True

        # create calculation graph with gradient on lm_head
        final_logits = actor(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
        print("final_logits = ", final_logits)
        distribution = Categorical(F.softmax(final_logits, dim=-1))
        print("distribution = ", distribution)

        # calculate values and returns
        for step in reversed(range(len(rewards))):
            last_value = rewards[step] + args.gamma * last_value
            returns.append(last_value)
        print("returns = ", returns)







        # calculate value
        state = input + ' ' + output
        state_ids = tokenizer(state, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
        print("state_ids = ", state_ids)
        value = critic(state_ids)
        print("value = ", value)

        #calculate


        for i in count():
            logits = get_logits(observation, args.nfirst).squeeze(0)[-1]
            print("logits = ", logits)

            dist = Categorical(F.softmax(logits))
            print("dist = ", dist)
            value = critic.forward(logits)

            action = dist.sample()
            print(action)
            next_observation, reward, done = env.step(action.squeeze(0).cpu().tolist())
            next_logits = get_logits(next_observation, args.nfirst).squeeze(0)[-1]
            print("next_logits = ", next_logits)
            next_value = critic.forward(next_logits)

            log_prob = dist.log_prob(action).unsqueeze(0)
            ret = args.gamma * next_value + reward
            adv = ret - value
            act_loss = -(log_prob * adv.detach()).sum()
            ctc_loss = adv.pow(2).sum()

            print("return = ", ret)
            print("reward = ", reward)
            print("advantage = ", adv)
            print("value = ", value)
            print("next_value = ", next_value)
            print("log_prob = ", log_prob)
            print("actor_loss = ", act_loss)
            print("critic_loss = ", ctc_loss)

            optimizerA.zero_grad()
            optimizerC.zero_grad()
            act_loss.backward(retain_graph=True)
            ctc_loss.backward()
            optimizerA.step()
            optimizerC.step()

            #log_probs.append(log_prob)
            #values.append(value)
            #rewards.append(torch.tensor([reward], dtype=torch.float, device=dvc))
            #masks.append(torch.tensor([1 - done], dtype=torch.float, device=dvc))

            observation = next_observation

            if done:
                print('Iteration: {}, Reward: {}'.format(iter, reward))
                break

        #logits = get_logits(next_observation, args.nfirst)
        #next_value = critic(logits)
        #returns = compute_returns(next_value, rewards, masks, args.gamma)

        #log_probs = torch.cat(log_probs)
        #returns = torch.cat(returns).detach()
        #values = torch.cat(values)

        #advantage = returns - values

        #actor_loss = -(log_probs * advantage.detach()).mean()
        #critic_loss = advantage.pow(2).mean()

        #optimizerA.zero_grad()
        #optimizerC.zero_grad()
        #actor_loss.backward()
        #critic_loss.backward()
        #optimizerA.step()
        #optimizerC.step()


if __name__ == "__main__":
    main()