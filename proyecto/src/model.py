import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=1024, n_layers=4, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                             dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden * 2, len(self.chars))

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_().to(device))
        return hidden

def save_checkpoint(net, opt, filename, train_history={}):
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'tokens': net.chars,
                  'train_history': train_history}
    torch.save(checkpoint, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location='cpu')
    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])
    return net, checkpoint

if __name__ == "__main__":
    print("Model definition is complete.")
