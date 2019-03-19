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
from Dataset import *
from Tool import *
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
    
    skip_dir = 'data/skip-thoughts'
    
    # load dictionary
    print('Loading dictionary...')
    with open(skip_dir+'/romance_clean.txt.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    dictionary =list(dictionary.keys())[0:20000-2]
    dictionary = ['EOS'] + dictionary + ['UNK']
    dict_size = len(dictionary)
    if 'dog' in dictionary:
        print('dog in dictionary')
    print('Dictionary successfully loaded.\n'+'*'*50)

    # build dataset and dataloader
    print('Building dataloader...')
    bs = config['batch_size']
    n_worker = config['num_worker']
    dataset = TextSet(file_dir=config['dir'], dictionary=dictionary)
    print('Total: {}'.format(len(dataset)))
    dataloader = utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=story_collate, num_workers=n_worker, pin_memory=True)
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
    with torch.no_grad():
        uniskip = UniSkip(skip_dir, dictionary).to(device)
    print('Skip-thoughts successfully loaded.\n'+'*'*50)
    
    # initialize
    print('Initializing...')
    
    model = Model(dict_size, 620, 2400, bs).to(device)
    
    weight = torch.tensor([1/(1+np.exp((dict_size-x)/dict_size)) for x in range(dict_size)], device=device)
    criterion = nn.NLLLoss(weight=weight, ignore_index=dict_size-1)
#     criterion = nn.NLLLoss(ignore_index=dict_size-1)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['patience'])
    
    Loss = []
    print('Initialization succeed.\n'+'*'*50)
    
    print('Start training...')
    model.train()
    for epoch in range(config['n_epoch']):
        for i, story in enumerate(dataloader):

            # formatting
            if bs == 1:
                story = torch.tensor(story)
                story_len = [len(story)]
            else:
                story, story_len = pad(story, dict_size-1)
            
            story = story.to(device)
            encoding = uniskip(story.view(bs, -1), lengths=story_len).detach().to(device) # input size: (batch_size, 2400)
            input = torch.cat((torch.zeros((bs, 1), device=device).long(), story[:, :-1]), dim=1)
            input = input.transpose(0, 1).unsqueeze(2).to(device)

            # backprop
            model.zero_grad()
            model.init_hidden(encoding.unsqueeze(0))           
            output, _ = model(input)
            loss = criterion(output, story.view(-1))
           
            loss.backward()
            optimizer.step()
            
            Loss.append(loss.item())
            
            scheduler.step(loss)
            
            if not i % 10:
                print(Loss[-1])
            
            
            # save model
            if not i % 2000:
                model_name = "epoch_{}-batch_{}-{}.pt".format(epoch, i, time.strftime("%Y%m%d-%H%M%S"))
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
#                 print(output)
                
            if not i % 10000:
                with open(result_path+'/loss_{}.pkl'.format(i), 'wb') as f:
                    pickle.dump(loss, f)



if __name__ == "__main__":
    main()