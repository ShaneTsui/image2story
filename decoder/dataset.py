import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from itertools import islice
from bisect import bisect_left
from re import findall

class TextSet(Dataset):

    def __init__(self, file_dir):
        self.dir = file_dir
        self.thr = 20
        self.sentence_end = ['.', '?', '!']
        self.sentence_cdf = []
        self.init_len()

    def __len__(self):
        return self.sentence_cdf[-1]
            
    def __getitem__(self, idx):
        assert idx < len(self)

        line_idx = bisect_left(self.sentence_cdf, idx) - 1

        text = [] # contain target corpus
        with open(self.dir, encoding="utf-8") as f:
            for l in islice(f, line_idx, bisect_left(self.sentence_cdf, idx+self.thr) - 1):
                text = text + findall('.*?[.!\?]', l)
                # story = story + l.split() + ['EOS']

        story = []
        sub_line_idx = idx-self.sentence_cdf[line_idx-1]
        for sent in text[sub_line_idx:sub_line_idx+self.thr]:
            story = story + sent.split() + ['EOS']
        sentence = story[:story.index('EOS')+1]

        return sentence, story

    def init_len(self):
        with open(self.dir, encoding="utf-8") as f:
            self.num_line = sum([self.count_sentence(l) for l in f]) - self.thr + 1

    def count_sentence(self, l):
        n_sentence = sum([1 for word in l.split() if word in self.sentence_end])
        self.sentence_cdf.append(n_sentence + sum(self.sentence_cdf))
        return n_sentence




class Onehot:

    def __init__(self, file_dir=None, dictonary=None):
        assert dile_dir or dictionary
        if file_dir:
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
        else:
            self.dict = dictionary

        self.num = len(self.dict)

    def __call__(self, words, return_idx=False):
        onehots = torch.zeros([len(words), self.num], dtype=torch.float)
        words = [self.filter(word) for word in words]
        idx = torch.LongTensor([self.dict.index(word) for word in words])
        if return_idx:
            return onehots.scatter_(1, idx.view(-1, 1), 1), idx
        return onehots.scatter_(1, idx.view(-1, 1), 1)

    def filter(self, word):
        if word not in self.dict:
            return 'UNK'
        return word

    def decode(self, onehot):
        val, idx = onehot.max(2)
        return self.dict(idx)




        