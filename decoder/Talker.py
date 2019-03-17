import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, dict_size, embed_size, hidden_size, dictionary=None):
        # input_size is the encodding dimension
        super(Model, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size =  hidden_size
        self.output_size = dict_size
        self.lstm = nn.GRU(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, dict_size, bias=True)
        # self.vec2word = nn.Linear(embed_size, dict_size, bias=True)
        # self.softmax = nn.Softmax(dim=2)
        self.embedding = nn.Embedding(dict_size, embed_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # self.dictionary = dictionary


    def init_hidden(self, encoding):
        self.hidden = encoding


    # def word2idx(self, text):
    #     return torch.LongTensor([self.dictionary.index(word) for word in text])


    def forward(self, input):
        embed = self.embedding(input)
        outvec, self.hidden = self.lstm(embed.view(-1, 1, self.embed_size), self.hidden)
        return self.softmax(self.linear(outvec.view(-1, self.hidden_size)))

