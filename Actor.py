import torch
import torch.nn.functional as F
import numpy as np


class Actor():
    def __init__(self, tokenizer, model, optimizer, device):
        self.vocab = tokenizer.get_vocab()
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        self.state_dim = len(self.vocab)
        self.action_dim = len(self.vocab)
        print('state_dim = {}'.format(self.state_dim))

        self.time_step = 0

    def choose_action(self, observation, device):
        cur_date, timelines = observation
        encoder_input = 
        decoder_input = timelines[cur_date]
        state = self.tokenizer(observation).input_ids
        state = torch.FloatTensor(state).to(device)
        # observation should be a contextual vector(?)
        network_output = self.model.forward(observation)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).numpy()
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
        return action

    def learn(self, state, action, td_error, device):
        self.time_step += 1
        # forward propagation
        state = torch.FloatTensor(state).to(device)
        softmax_input = self.model.forward(state).unsqueeze(0)
        action = torch.LongTensor([action]).to(device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')
        
        # backward propagation
        loss_a = neg_log_prob * td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()
