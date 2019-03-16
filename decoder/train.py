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

    # import encoder
    # encoder = Encoder()
    # encoder.load_state_dict(torch.load(config['encoder_dir']))

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
    
#     with open(skip_dir+'/dictionary_text.txt', 'wb') as f:
#         for word in dictionary:
#             w = word+'\n'
#             f.write(w.encode("utf-8"))


#     dictionary = []
#     with open(skip_dir+'/dictionary.txt', encoding="utf-8") as f:
#         for l in f:
#             dictionary.append(l)
            
#     D = np.load(skip_dir+'/utable.npy')[0:20000]
    
    # read vocab
    print('Reading vocabulary in the corpus...')
    if not config['vocab']:
        with open(config['dir'], encoding="utf-8") as g:
            book = g.read()
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
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    print('Vocabulary successfully read.\n'+'*'*50)
#     vocab = [unicodedata.normalize('NFKD', v).encode('ascii','ignore') for v in vocab]
#     vocab = [v.encode("utf-8") for v in list(set(vocab))]
#     print(type(vocab[0]))
    
    # load skipvector
    print('Loading skip-thoughts...')
    uniskip = UniSkip(skip_dir, vocab)
    print('Skip-thoughts successfully loaded.\n'+'*'*50)
    
    # initialize
#     onehot = Onehot(config['dir'])
#     model = Talker(onehot.num, 2400)
    onehot = Onehot(dictionary=dictionary)
#     eos_embedding = -torch.ones((1, 620))
    eos_hot = onehot(['EOS'])
    model = Talker(620, 2400, len(dictionary))
    model.to(device)
#     criterion = nn.CrossEntropyLoss()
    
    criterion = nn.MSELoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    word_embedding = nn.Embedding(20000, 620)
#     word_embedding.to(device)

    Loss = []
    
    print('Start training...')
    for epoch in range(config['n_epoch']):
        for i, (sentence, story) in enumerate(dataloader):

            # formatting
            sentence, sentence_idx = onehot([w[0] for w in sentence], return_idx=True)
            story, story_idx = onehot([w[0] for w in story], return_idx=True)
            target_vec = word_embedding(story_idx)
            in_vec = word_embedding(torch.cat((torch.LongTensor([0]), story_idx[1:]), dim=0))
            in_vec = in_vec.unsqueeze(1)
            
            encoding = uniskip(sentence_idx.view(1, -1), lengths=[len(sentence_idx)])
            
            # to device         
            encoding = encoding.to(device)
            sentence = sentence.to(device)
            story = story.to(device)
            target_vec = target_vec.to(device)
            in_vec = in_vec.to(device)

            # backprop
            model.init_hidden(encoding.view(1, 1, -1))
            optimizer.zero_grad()
            out_vec, output = model(in_vec)
#             print(target_vec.shape)
#             print(output.shape)
#             print(out_vec.shape)
#             print(story.shape)
#             loss = criterion(output, Variable(story.long()))
            loss = criterion(out_vec, target_vec.view(-1, 1, 620))
            
            loss.backward()
            optimizer.step()
            
            
            Loss.append(loss.item())
            
            print(Loss[-1])
            
            
            # save model
            model_name = "epoch_{}-batch_{}-{}.pt".format(epoch, i, time.strftime("%Y%m%d-%H%M%S"))
            torch.save(model.state_dict(), os.path.join(model_path, model_name))



if __name__ == "__main__":
    main()