import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import islice

class TextSet(Dataset):

    def __init__(self, file_dir):
        self.dir = file_dir
        self.thr = 10
        self.init_len()

    def __len__(self):
        return self.num_line
            
    def __getitem__(self, idx):
        assert idx < len(self)
        story = []
        with open(self.dir) as f:
            for l in islice(f, idx, idx+self.thr):
                story = story + l.split() + ['EOS']
        sentence = story[:story.index('EOS')+1]
        return sentence, story

    def init_len(self):
        with open(self.dir) as f:
            self.num_line = sum([1 for l in f]) - self.thr + 1


class Onehot:

    def __init__(self, file_dir):
        self.dict = []

        with open(file_dir) as f:
            book = f.readlines()

        for line in book:
            line = line.strip().split()
            for word in line:
                if word not in self.dict:
                    self.dict.append(word)

        if 'EOS' not in book:
            self.dict.append('EOS')

        self.num = len(self.dict)

    def __call__(self, words, return_idx=False):
        onehots = torch.zeros([len(words), self.num], dtype=torch.float)
        idx = torch.LongTensor([self.dict.index(word) for word in words])
        if return_idx:
            return onehots.scatter_(1, idx.view(-1, 1), 1), idx
        return onehots.scatter_(1, idx.view(-1, 1), 1)
        