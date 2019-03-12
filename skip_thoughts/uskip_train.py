import os
import sys
sys.path.insert(0,'/datasets/home/home-02/60/960/kshi/image2story')
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from skip_thoughts.data_loader import DataLoader
from skip_thoughts.model import UniSkip
from skip_thoughts.config import *
from datetime import datetime, timedelta
import time
import pathlib

if __name__ == '__main__':
    # Specify paths
    Root = '/datasets/home/home-02/60/960/kshi/image2story'
    data_path = os.path.join(Root,'data/books_large_p1.txt')
    # data_dummy_path = os.path.join(Root,'data/dummy_corpus.txt')
    logs_path = os.path.join(Root,'logs')

    # Build data loader
    d = DataLoader(data_path)

    #Build directory to save loss in files for futher usage.
    loss_path = Root + '/logs/losses/{}/'.format(time.strftime("%Y%m%d-%H%M%S"))
    loss_pathlib = pathlib.Path(loss_path)
    if not loss_pathlib.exists():
        pathlib.Path(loss_pathlib).mkdir(parents=True, exist_ok=True)


    # create model
    lr = 1e-4
    model = UniSkip()
    model = model.to(computing_device)
    optimizer = optim.Adam(model.parameters(),lr =lr)
    # load weights
    load_pretrained = False
    if load_pretrained: 
        MODEL_PATH = os.path.join(logs_path, 'skip-best-2400')
        model.load_state_dict(torch.load(MODEL_PATH))


    loss_trail = []
    last_best_loss = None
    current_time = datetime.utcnow()

    def debug(i, loss, prev, nex, prev_pred, next_pred):
        global loss_trail
        global last_best_loss
        global current_time

        this_loss = loss.data[0]
        loss_trail.append(this_loss)
        loss_trail = loss_trail[-20:]
        new_current_time = datetime.utcnow()
        time_elapsed = str(new_current_time - current_time)
        current_time = new_current_time
        print("Iteration {}: time = {} last_best_loss = {}, this_loss = {}".format(
                  i, time_elapsed, last_best_loss, this_loss))

        print("prev = {}\nnext = {}\npred_prev = {}\npred_next = {}".format(
            d.convert_indices_to_sentences(prev),
            d.convert_indices_to_sentences(nex),
            d.convert_indices_to_sentences(prev_pred),
            d.convert_indices_to_sentences(next_pred),
        ))
        #Save loss in a txt file.
        with open(os.path.join(loss_path, "training.txt"), "a") as f:
            f.write(str(this_loss.item()) +"\n")

        try:
            trail_loss = sum(loss_trail)/len(loss_trail)
            if last_best_loss is None or last_best_loss > trail_loss:
                print("Loss improved from {} to {}".format(last_best_loss, trail_loss))

                save_loc = logs_path +'/skip-best-2400'.format(lr, VOCAB_SIZE)
                print("saving model at {}".format(save_loc))
                torch.save(model.state_dict(), save_loc)

                last_best_loss = trail_loss
        except Exception as e:
            print("Couldn't save model because {}".format(e))


    print("Starting training...")
    # Roughly 312500 iterations one epoch if batch size is 128.
    for i in range(0, 100000):
        optimizer.zero_grad()
        sentences, lengths = d.fetch_batch(128,i)
        loss, prev, nex, prev_pred, next_pred  = model(sentences, lengths)
        if i % 20 == 0:
            debug(i, loss, prev, nex, prev_pred, next_pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)
        optimizer.step()
     
    idx = i
    for g in optimizer.param_groups:
        g['lr'] = lr/10
        
    for i in range(idx, 200000):
        optimizer.zero_grad()
        sentences, lengths = d.fetch_batch(128,i)
        loss, prev, nex, prev_pred, next_pred  = model(sentences, lengths)
        if i % 20 == 0:
            debug(i, loss, prev, nex, prev_pred, next_pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)
        optimizer.step()
        
        
    for g in optimizer.param_groups:
        g['lr'] = lr/100
    idx = i  
    for i in range(idx, 100000):
        optimizer.zero_grad()
        sentences, lengths = d.fetch_batch(128,i)
        loss, prev, nex, prev_pred, next_pred  = model(sentences, lengths)
        if i % 20 == 0:
            debug(i, loss, prev, nex, prev_pred, next_pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)
        optimizer.step() 