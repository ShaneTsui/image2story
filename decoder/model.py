import torch
import torch.nn as nn


class Talker(nn.Module):

    def __init__(self, dict_size, hidden_size):
        # input_size is the encodding dimension
        super(Talker, self).__init__()
        
        self.input_size = dict_size
        self.hidden_size =  hidden_size
        self.output_size = dict_size
        self.lstm = nn.LSTM(dict_size, hidden_size)
        self.linear = nn.Linear(hidden_size, dict_size, bias=True)
        self.softmax = nn.Softmax(dim=2)


    def init_hidden(self, encoding):
        self.hidden = encoding


    def forward(self, input):
        output, self.hidden = self.lstm(input, self.hidden)
        return self.softmax(self.linear(output))