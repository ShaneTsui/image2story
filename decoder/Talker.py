import torch
import torch.nn as nn


class Talker(nn.Module):

    def __init__(self, embed_size, hidden_size, dict_size):
        # input_size is the encodding dimension
        super(Talker, self).__init__()
        
        self.input_size = embed_size
        self.hidden_size =  hidden_size
        self.output_size = embed_size
        self.lstm = nn.GRU(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, embed_size, bias=True)
        self.vec2word = nn.Linear(embed_size, dict_size, bias=True)
        self.softmax = nn.Softmax(dim=2)


    def init_hidden(self, encoding):
        self.hidden = encoding


    def forward(self, input):
        output, self.hidden = self.lstm(input, self.hidden)
        vec = self.linear(output)
        return vec, self.softmax(self.vec2word(vec))

