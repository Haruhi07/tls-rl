import torch
import torch.nn.functional as F
import numpy as np
from utils import nfirst, format_decoder_input


class Actor():
    def __init__(self, tokenizer, model, optimizer, device, nfirst=5):
        self.vocab = tokenizer.get_vocab()
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.nfirst = nfirst

        self.state_dim = self.tokenizer.vocab_size
        self.action_dim = self.tokenizer.vocab_size
        print('state_dim = {}'.format(self.state_dim))

        self.time_step = 0

    def choose_action(self, observation, device):
        cluster, timeline = observation
        encoder_input = [nfirst(a.text, self.nfirst) for a in cluster.articles]
        print(encoder_input)
        decoder_input = format_decoder_input(timeline["text"])
        print(decoder_input)

        encoder_input_ids = self.tokenizer(encoder_input, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        decoder_input_ids = self.tokenizer(decoder_input, return_tensors="pt").input_ids.to(device)
        
        lm_logits = self.model.forward(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids).logits
        print(lm_logits)
        with torch.no_grad():
            prob = F.softmax(lm_logits, dim=2).detach().cpu().numpy()
        prob = prob.reshape(-1)
        print(prob.shape)
        action = np.random.choice(range(prob.shape[0]), p=prob)
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
