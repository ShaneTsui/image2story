import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.autograd import Variable

import numpy as np
import sys
import yaml
import os
import time
import pathlib
import pickle

from Talker import *
from TextDataset import *
from tool import *
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
from itertools import islice

def main():
    # load configuration
    cwd = os.getcwd()
    with open('config.yaml') as f:
        config = yaml.load(f)
        
    skip_dir = 'data/skip-thoughts'

    # load dictionary
    print('Loading dictionary...')
    with open(skip_dir+'/romance_clean.txt.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    dictionary =list(dictionary.keys())[0:20000-2]
    dictionary = ['EOS'] + dictionary + ['UNK']
    print('Dictionary successfully loaded.\n'+'*'*50)

    # read vocab
    print('Reading vocabulary in the corpus...')
    if not config['vocab']:
        with open(config['dir'], encoding="utf-8") as g:
            book = g.read()
            vocab = book.split()
            if 'EOS' not in vocab:
                vocab.append('EOS')
        vocab = list(set(vocab))
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    print('Vocabulary successfully read.\n'+'*'*50)
    
    # load skipvector
    print('Loading skip-thoughts...')
    uniskip = UniSkip(skip_dir, vocab)
    print('Skip-thoughts successfully loaded.\n'+'*'*50)
    
    # check GPU
    assert torch.cuda.is_available(), 'CUDA is not available'
    device = torch.device('cuda')


    onehot = Onehot(dictionary=dictionary)
    # eos_hot = onehot(['EOS'])
    model = Talker(620, 2400, len(dictionary))
    model.load_state_dict(torch.load(config['model_path']))
    model.to(device)
    model.eval()

    word_embedding = nn.Embedding(20000, 620)

    sentence = ['I', 'love', 'my', 'teacher', 'and', 'his', 'dog', '.', 'EOS']
    sentence_onehot, sentence_idx = onehot([w[0] for w in sentence], return_idx=True)
    encoding = uniskip(sentence_idx.view(1, -1), lengths=[len(sentence_idx)])
    model.init_hidden(encoding)
    voice = Voice(dictionary=dictionary, embedding=word_embedding)

#     n_sentence = 2
    n_word = 10
    neuralstory = []
    n_st = 0

    # to device
    encoding = encoding.to(device)

    model.init_hidden(encoding.view(1, 1, -1))

    c = word_embedding(torch.LongTensor([0]))
    
    print(voice.dict_vec[0:20])
    
    for k in range(n_word):
#         print('input shape', c.shape)
#         print(c)
        c = c.to(device)
        outvec, output = model(c.view(1, 1, 620))
#         print('output shape', outvec.shape)
        outvec = outvec.to('cpu')
        word = voice(outvec)
#         print('word', word)
        neuralstory.append(word) 
        c = word_embedding(torch.LongTensor([dictionary.index(word)]))
#         if word == 'EOS':
#             n_st = n_st + 1
    
    

#     while n_st <= n_sentence:
# #         print('input shape', c.shape)
#         c = c.to(device)
#         outvec, output = model(c.view(1, 1, 620))
# #         print('output shape', outvec.shape)
#         outvec = outvec.to('cpu')
#         word = voice(outvec)
#         print('word', word)
#         neuralstory.append(word) 
#         c = word_embedding(torch.LongTensor([dictionary.index(word)]))
#         if word == 'EOS':
#             n_st = n_st + 1
    
    print(sentence)
    print(neuralstory)
    
#     print(list(model.linear.parameters()))
    
    

    

if __name__ == "__main__":
    main()