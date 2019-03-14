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
import unicodedata

from Talker import *
from TextDataset import *
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
from itertools import islice


def main():

    # load configuration
    cwd = os.getcwd()
    with open('config.yaml') as f:
        config = yaml.load(f)

    # models and results saving setup
    ct = time.strftime('%m/%d/%Y-%H:%M:%S')

    model_path = '{}/models/{}'.format(cwd, ct)
    model_pathlib = pathlib.Path(model_path)
    if not model_pathlib.exists():
        model_pathlib.mkdir(parents=True, exist_ok=True)

    result_path = '{}/results/{}'.format(cwd, ct)
    result_pathlib = pathlib.Path(result_path)
    if not result_pathlib.exists():
        result_pathlib.mkdir(parents=True, exist_ok=True)

    # check GPU
    assert torch.cuda.is_available(), 'CUDA is not available'
    device = torch.device('cuda')

    # import encoder
    # encoder = Encoder()
    # encoder.load_state_dict(torch.load(config['encoder_dir']))

    # build dataset and dataloader
    print(5)
    dataset = TextSet(file_dir=config['dir'])
    dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    
    # load skipvector
    print(1)
    skip_dir = 'data/skip-thoughts'
    
    # load dictionary
    print(2)
    dictionary = []
    with open(skip_dir+'/dictionary.txt', encoding="utf-8") as f:
        for l in f:
            dictionary.append(l)
    
    # read vocab
    print(3)
#     vocab = []
    with open(config['dir'], encoding="utf-8") as g:
        book = g.read()
        book = unicodedata.normalize('NFC', book).encode('ascii','ignore')
        vocab = book.split()
#         book = g.readlines()
#         for line in book:
#             line = line.strip()
#             for word in line.split():
#                 if word not in vocab:
#                     vocab.append(word)
        if 'EOS' not in vocab:
            vocab.append('EOS')
    vocab = list(set(vocab))
#     vocab = [unicodedata.normalize('NFKD', v).encode('ascii','ignore') for v in vocab]
#     vocab = [v.encode("utf-8") for v in list(set(vocab))]
#     print(type(vocab[0]))
    
    print(4)
    uniskip = UniSkip(skip_dir, vocab)
    
    # initialize
#     onehot = Onehot(config['dir'])
#     eos_hot = onehot(['EOS'])
#     model = Talker(onehot.num, 2400)
    print(6)
    onehot = Onehot(dictionary=dictionary)
    model = Talker(len(dictionary), 2400)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    

    for epoch in range(config['n_epoch']):
        for i, (sentence, story) in enumerate(dataloader):

            # formatting
            sentence, sentence_idx = onehot([w[0] for w in sentence], return_idx=True)
            story, story_idx = onehot([w[0] for w in story], return_idx=True)
            story.to(device), story_idx.to(device)
            encoding = uniskip(sentence_idx, lengths=[len(sentence_idx)])
            model.init_hidden(encoding)
            input = torch.cat((eos_hot, story[1:, :]), dim=0)
            
            # to device
            sentence.to(device)
            story_idx.to(device)
            input.to(device)

            # backprop
            optimizer.zero_grad()
            output = model(input.unsqueeze(1))
#             print('target shape', story.unsqueeze(1).shape)
#             print('output shape', output.shape)
            loss = criterion(output.squeeze(1), Variable(story_idx.long()))
            # loss = criterion(output, story.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            # save model
            model_name = "epoch_{}-batch_{}-{}.pt".format(epoch, i, time.strftime("%Y%m%d-%H%M%S"))
            torch.save(model.state_dict(), os.path.join(model_path, model_name))



if __name__ == "__main__":
    main()