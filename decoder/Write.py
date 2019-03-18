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
from Tool import *
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
    dict_size = len(dictionary)
    print('Dictionary successfully loaded.\n'+'*'*50)
    
    with open('fuck.txt', 'wb') as f:
        b = '\n'.join(dictionary)
        f.write(b.encode('utf-8'))

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
    with torch.no_grad():
        uniskip = UniSkip(skip_dir, vocab)
    print('Skip-thoughts successfully loaded.\n'+'*'*50)
    
    # check GPU
    assert torch.cuda.is_available(), 'CUDA is not available'
    device = torch.device('cuda')

    onehot = Onehot(dictionary=dictionary)
    model = Model(dict_size, 620, 2400, 1).to(device)
    model.load_state_dict(torch.load(config['model_path']))
    model.to(device)
    model.eval()

    sentence = ['I', 'love', 'my', 'teacher', 'and', 'his', 'dog', '.', 'EOS']
    sentence_onehot, sentence_idx = onehot([w[0] for w in sentence], return_idx=True)
    encoding = uniskip(sentence_idx.view(1, -1), lengths=[len(sentence_idx)]).to(device)
    
    n_word = 20
    neuralstory = []
    n_st = 0

    
    word_id = torch.LongTensor([5])
    
    
    for k in range(n_word):
        model.init_hidden(encoding.view(1, 1, -1))

        word_id = word_id.to(device)
        output, encoding = model(word_id)
        word_id = torch.argmax(output)
        word = dictionary[word_id.item()]
        neuralstory.append(word) 
        
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
    with open('Neuralstory.txt', 'wb') as f:
        for sentence_word in sentence:
            f.write(sentence_word.encode('utf-8'))
        f.write('\n'.encode('utf-8'))
        for story_word in neuralstory:
            f.write(story_word.encode('utf-8'))
    
#     print(list(model.linear.parameters()))
    
    

    

if __name__ == "__main__":
    main()