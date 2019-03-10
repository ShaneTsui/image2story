import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.autograd import Variable

import numpy as np
import yaml
import os
import time
import pathlib
from itertools import islice


from model import *
from dataset import *



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
    dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # initialize
    onehot = Onehot(config['dir'])
    eos_hot = onehot(['EOS'])
    model.to(device)
    model = Talker(onehot.num, 20)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    

    for epoch in range(config['n_epoch']):
        for i, sample in enumerate(dataloader):

            # formatting
            sentence = onehot([w[0] for w in sentence])
            story, story_idx = onehot([w[0] for w in story], return_idx=True)
            story.to(device), story_idx.to(device)
            encoding = encoder(sentence)
            model.init_hidden(encoding)
            input = torch.cat((eos_hot, story[1:, :]), dim=0)

            # to device
            sentence.to(device)
            story_idx.to(device)
            input.to(device)

            # backprop
            model.zero_grad()
            output = model(input.unsqueeze(1))
            loss = criterion(output.squeeze(1), Variable(story_idx.long()))
            # loss = criterion(output, story.unsqueeze(1))
            loss.backward()
            optimizer.step()



if __name__ == "__main__":
    main()