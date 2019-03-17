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

    # build dataset and dataloader
    dataset = TextSet(file_dir=config['dir'])
    dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    
    
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
    with torch.no_grad():
        uniskip = UniSkip(skip_dir, vocab)
    print('Skip-thoughts successfully loaded.\n'+'*'*50)
    
    # initialize
    onehot = Onehot(dictionary=dictionary)

    model = Model(len(dictionary), 620, 2400)
    model.to(device)
    
    weight = [1/(1+np.exp((10000-x)/5000)) for x in range(20000)]
    criterion = nn.NLLLoss(weight=torch.tensor(weight), ignore_index=19999)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)

    Loss = []
    
    print('Start training...')
    for epoch in range(config['n_epoch']):
        for i, (sentence, story) in enumerate(dataloader):

            # formatting
#             sentence_idx = word2idx(sentence, dictionary)
#             story_idx = word2idx(story, dictionary)
            sentence, sentence_idx = onehot([w[0] for w in sentence], return_idx=True)
            story, story_idx = onehot([w[0] for w in story], return_idx=True)
            encoding = uniskip(sentence_idx.view(1, -1), lengths=[len(sentence_idx)])
            input = torch.cat((torch.LongTensor([0]), story_idx[0:-1]), dim=0)
#             print(sentence_idx)
#             print(input)

            # to device
            # sentence = sentence.to(device)
            # story = story.to(device)
            input = input.to(device)
            story_idx = story_idx.to(device)
            encoding = encoding.to(device)

            # backprop
            model.zero_grad()
            model.init_hidden(encoding.view(1, 1, -1))           
            output = model(input)
            loss = criterion(output, story_idx)
            
            loss.backward()
            optimizer.step()           
            
            Loss.append(loss.item())
            
            scheduler.step(loss)
            
            print(Loss[-1])
            
            
            # save model
            if not i % 50:
                model_name = "epoch_{}-batch_{}-{}.pt".format(epoch, i, time.strftime("%Y%m%d-%H%M%S"))
                torch.save(model.state_dict(), os.path.join(model_path, model_name))
#                 print(output)
                
            if not i % 1000:
                with open(result_path+'/loss_{}.pkl'.format(i), 'wb') as f:
                    pickle.dump(loss, f)



if __name__ == "__main__":
    main()