from argparse import ArgumentParser
from env_utils import extract_keywords
from environment import Environment
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from Actor import Actor
from Critic import Critic
import os
import pathlib
import torch
import json
import pickle


def setup_env(args):
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
    env = Environment(clusters, keywords, length)
    return env

def main():
    parser = ArgumentParser()
    # Configuration
    parser.add_argument("--dataset", type=str, required=True)
    # RL
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()

    # Real Actor network
    model_name = 'google/pegasus-multi_news'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    actor = Actor(tokenizer, model, optimizer, device)
    print("actor initialized...")
    critic = Critic(actor.state_dim, device, args)
    print("critic initialized...")
    env = setup_env(args)
    print("env initialized...")

    for episode in range(args.episode):
        # initialize task
        env.reset()
        state = env.observation()
        #Train
        for step in range(args.max_length):
            print("observation = ", state)
            action = actor.choose_action(state, device) # action should be the index of words, which means selecting this word
            print("action = ", action)
            next_state, reward, done = env.step(action)
            td_error = critic.train_Q_network(state, reward, next_state)
            actor.learn(state, action, td_error)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(args.test_size):
                state = env.reset()
                for j in range(args.max_length):
                    action = actor.choose_action(state)  # direct action for test
                    state, reward, done = env.step(action)
                    total_reward += reward
                    if done:
                        break

            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)

if __name__ == "__main__":
    main()