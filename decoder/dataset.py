import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import islice
from bisect import bisect_left
from re import findall

import pickle as pkl
import decoder.config as config
    
class TextSet(Dataset):

    def __init__(self, file_dir, dictionary, n_words):
        # dictionary is a list
        self.dir = file_dir
        self.thr = 5
        self.dictionary = dictionary
        self.n_words = n_words
        self.init()

    def __len__(self):
        return len(self.book)
            
    def __getitem__(self, idx):
        
        # text = findall('.*?[.!\?-]', self.book[idx])
        # print(text[0])
        
        # story = []
        # for sent in self.book[idx]:
        #     story = story + sent.split() + ['<eos>']
        # if len(story) == 0:
        #     story = self[idx+1]
        
        # list of words
        return self.encode(self.book[idx].split() + ['<eos>'])

    def init(self):
        # with open(self.dir, encoding="utf-8") as f:
        #     self.book = f.readlines()
        # self.book = [l.strip('\n') for l in self.book if 30<len(l.split())<=80]
        with open(config.paths['text'], 'rb') as f:
            print('loading text')
            text = pkl.load(f)
        self.book = text
        

    def encode(self, text):
        encoding = []
        for word in text:
            if word in self.dictionary and self.dictionary[word] <= self.n_words:
                encoding.append(self.dictionary[word])
            else:
                encoding.append(self.dictionary['UNK'])
        return encoding


class Onehot:

    def __init__(self, file_dir=None, dictionary=None):
        assert file_dir or dictionary
        if file_dir:
            self.dict = []

            with open(file_dir) as f:
                book = f.readlines()

            for line in book:
                line = line.strip()
                for word in line.split():
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
