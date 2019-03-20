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

from decoder.talker import *
from decoder.dataset import *
from decoder.tool import *
sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
from itertools import islice

import decoder.config as decoder_config

def main():

    # load configuration
    cwd = os.getcwd()
    with open('config.yaml') as f:
        config = yaml.load(f)

    # models and results saving setup
    ct = time.strftime("%Y%m%d-%H%M%S")

    model_path = '{}/results/{}/models'.format(cwd, ct)
    model_pathlib = pathlib.Path(model_path)
    if not model_pathlib.exists():
        model_pathlib.mkdir(parents=True, exist_ok=True)

    result_path = '{}/results/{}/results'.format(cwd, ct)
    result_pathlib = pathlib.Path(result_path)
    if not result_pathlib.exists():
        result_pathlib.mkdir(parents=True, exist_ok=True)

    # check GPU
    assert torch.cuda.is_available(), 'CUDA is not available'
    device = torch.device('cuda')
    
    skip_dir = '../data/skip-thoughts'
    
    # load dictionary
    print('Loading dictionary...')
    with open(decoder_config.paths['dictionary'], 'rb') as f:
        dictionary = pickle.load(f)

    # vocabulary =list(dictionary.keys())[0:20000-2]
    # vocabulary = ['EOS'] + vocabulary + ['UNK']
    dict_size = decoder_config.settings['decoder']['n_words']
    # if 'dog' in vocabulary:
    #     print('dog in dictionary')
    print('Dictionary successfully loaded.\n'+'*'*50)

    # build dataset and dataloader
    print('Building dataloader...')
    batch_size = config['batch_size']
    n_worker = config['num_worker']

    # Load data
    dataset = TextSet(file_dir=config['dir'], dictionary=dictionary, n_words=dict_size)
    print('Total: {}'.format(len(dataset)))
    dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=story_collate, num_workers=n_worker, pin_memory=True)
    print('Dataloader successfully built.\n'+'*'*50)
    
#     # read vocab
#     print('Reading vocabulary in the corpus...')
#     if not config['vocab']:
#         with open(config['dir'], encoding="utf-8") as g:
#             book = g.read()
#             vocab = book.split()
#             if 'EOS' not in vocab:
#                 vocab.append('EOS')
#         vocab = list(set(vocab))
#         with open('vocab.pkl', 'wb') as f:
#             pickle.dump(vocab, f)
#     else:
#         with open('vocab.pkl', 'rb') as f:
#             vocab = pickle.load(f)
#     print('Vocabulary successfully read.\n'+'*'*50)

    # load skipvector
    print('Loading skip-thoughts...')
    uniskip = UniSkip(skip_dir, list(dictionary.keys())).to(device)
    uniskip.eval()
    print('Skip-thoughts successfully loaded.\n'+'*'*50)
    
    # initialize
    print('Initializing...')
    
    model = Decoder(vocab_size=decoder_config.settings['decoder']['n_words'], \
                    embed_size=decoder_config.settings['decoder']['dim_word'], \
                    hidden_size=decoder_config.settings['decoder']['dim']).to(device)
    
    # weight = torch.tensor([1/(1+np.exp((dict_size-x)/dict_size)) for x in range(dict_size)], device=device)
    # criterion = nn.NLLLoss(weight=weight, ignore_index=dict_size-1)
    criterion = nn.NLLLoss(ignore_index=dictionary['UNK'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['patience'])

    log_softmax = nn.LogSoftmax(dim=2)

    Loss = []
    print('Initialization succeed.\n'+'*'*50)
    
    print('Start training...')
    model.train()
    for epoch in range(config['n_epoch']):
        for i, story in enumerate(dataloader):
            # formatting
            if batch_size == 1:
                story = torch.tensor(story)
                story_len = [len(story)]
            else:
                story, story_len = pad(story, dictionary['UNK'])
            
            story = story.to(device)
            with torch.no_grad():
                encoding = uniskip(story.view(batch_size, -1), lengths=story_len).detach().to(device) # input size: (batch_size, 2400)
            input = torch.cat((torch.zeros((batch_size, 1), device=device).long(), story[:, :-1]), dim=1)
            # input = input.transpose(0, 1).to(device)

            # backprop
            model.zero_grad()
            hidden = model.init_hidden(encoding.unsqueeze(0))
            output, hidden = model(input, hidden)
            logits = log_softmax(output)
            loss = criterion(logits.view(-1, dict_size), story.view(-1))
           
            loss.backward()
            optimizer.step()
            
            Loss.append(loss.item())
            
            # scheduler.step(loss)
            
            if not i % 10:
                print(Loss[-1])
            
            
            # save model
            if not i % 100:
                model_name = "epoch_{}-batch_{}-loss_{}-{}.pt".format(epoch, i, loss.item(), time.strftime("%Y%m%d-%H%M%S"))
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
#                 print(output)
                
            if not i % 100:
                with open(result_path+'/loss_{}.pkl'.format(i), 'wb') as f:
                    pickle.dump(loss, f)



if __name__ == "__main__":
    main()