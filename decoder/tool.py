import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from itertools import islice
from bisect import bisect_left
from re import findall

class Voice:

    def __init__(self, dictionary, embedding):
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        self.embedding = embedding
        self.dict2vec()

    def __call__(self, vec):
#         print('dict shape',self.dict_vec.shape)
        print(self.dict_vec[0:5, 0:5])
#         print(vec.shape)
        print(vec.squeeze(0)[0 ,0:5])
        dist = torch.norm(self.dict_vec-vec.squeeze(0), dim=1)
#         print('distance shape', dist.shape)
#         print(dist[0:20])
        idx = torch.argmin(dist, dim=0)
        print('idx', idx)
        return self.dictionary[idx]


    def dict2vec(self):
        dict_vec = self.embedding(torch.LongTensor([range(self.dict_size)]))
        self.dict_vec = dict_vec.squeeze(0)

def story_collate(batch):
    return batch
        
def pad(batch, val):
    # batch: a list of tensor
    story_len = [len(story) for story in batch]
    max_len = max(story_len)
    story_pad = lambda story, l: torch.cat((torch.LongTensor(story), val*torch.ones(max_len-l).long()))
    story = [story_pad(story, l) for story, l in zip(batch, story_len)]
    # output size: (batch_size, max_len)
    return torch.stack(story), story_len
