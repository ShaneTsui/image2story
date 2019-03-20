import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import sys
import yaml

import os
import time
import pathlib
import pickle

from decoder.talker import *
from decoder.dataset import *
from decoder.tool import *
# sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
import decoder.config as decoder_config
from itertools import islice

def main():
    # load configuration
    cwd = os.getcwd()
    with open('config.yaml') as f:
        config = yaml.load(f)
        
    skip_dir = '../data/skip-thoughts'

    # load dictionary
    print('Loading dictionary...')
    with open(decoder_config.paths['dictionary'], 'rb') as f:
        dictionary = pickle.load(f)
    dict_size = decoder_config.settings['decoder']['n_words']
    print('Dictionary successfully loaded.\n' + '*' * 50)
    
    # load skipvector
    print('Loading skip-thoughts...')

    uniskip = UniSkip(skip_dir, list(dictionary.keys()))
    print('Skip-thoughts successfully loaded.\n'+'*'*50)
    
    # check GPU
    assert torch.cuda.is_available(), 'CUDA is not available'
    device = torch.device('cuda')
    
    # load model
    model = Decoder(vocab_size=decoder_config.settings['decoder']['n_words'], \
                    embed_size=decoder_config.settings['decoder']['dim_word'], \
                    hidden_size=decoder_config.settings['decoder']['dim']).to(device)
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()

    sentence = ['I', 'like', 'my', 'teacher', 'and', 'his', 'dog', '.', '<eos>']
    sentence_idx = Variable(torch.LongTensor([dictionary[w] if w in dictionary else dictionary['UNK'] for w in sentence]))
    print(sentence_idx)
    encoding = uniskip(sentence_idx.view(1, -1)).to(device)
    
    n_word = 20
    neuralstory = []
    n_st = 5

    
    word_id = torch.LongTensor([0])

    # 1: <eos>, 2: <unk>, 0: end_padding
    word_idict = dict()
    for kk, vv in dictionary.items():
        word_idict[vv] = kk
    word_idict[-1] = 'hehehe'


    k = 0
    temperature = 0.7
    model.init_hidden(encoding.unsqueeze(0))
    while k < 200 :
        # print(encoding)
        word_id = word_id.to(device)
        output, encoding = model(word_id.view(1, -1))
        # print(output[0:6])
        prob = F.softmax(output.view(-1) / temperature)
        word_id = torch.multinomial(prob, 1)
        # word_id = torch.argmax(output) + 1
        # print(word_id.item())
        word = word_idict[word_id.item()] # dictionary[word_id.item()]
        neuralstory.append(word)
        k = k + 1

    print(" ".join(neuralstory))
    # with open('Neuralstory.txt', 'wb') as f:
    #     sentence = [word+' ' for word in sentence]
    #     for sentence_word in sentence:
    #         f.write(sentence_word.encode('utf-8'))
    #     f.write('\n'.encode('utf-8'))
    #     for story_word in neuralstory:
    #         f.write(story_word.encode('utf-8'))
    
#     print(list(model.linear.parameters()))
    
    

    

if __name__ == "__main__":
    main()