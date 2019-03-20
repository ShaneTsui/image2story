import torch
from torch.autograd import Variable
import sys
import os
from tqdm import tqdm

sys.path.append('skip-thoughts.torch/pytorch')
from skipthoughts import UniSkip
import decoder.config as decoder_config

import pickle
import numpy as np
from decoder.tool import *

# check GPU
assert torch.cuda.is_available(), 'CUDA is not available'
device = torch.device('cuda')

# load dictionary
with open('../decoder/' + decoder_config.paths['dictionary'], 'rb') as f:
    dictionary = pickle.load(f)

# load skipvector
skip_dir = '../data/skip-thoughts'
uniskip = UniSkip(skip_dir, list(dictionary.keys())).to(device)
uniskip.eval()

# # load captions
dir = '../data/'
# with open(dir + 'coco2014-vgg/coco_train_caps.txt') as f:
#     train_captions = f.readlines()
#
# with open(dir + 'coco2014-vgg/coco_dev_caps.txt') as f:
#     dev_captions = f.readlines()
#
# captions = train_captions + dev_captions
#
# captions = [Variable(
#     torch.LongTensor([dictionary[word] if word in dictionary.keys() else dictionary['UNK'] for word in l.split()])) for
#             l in captions]
# # cap, cap_len = pad(captions, 0)
# with torch.no_grad():
#     cap_st_vec_total = torch.zeros(1, 2400).to(device)
#     for cap in tqdm(captions):
#         cap = cap.to(device)
#         cap_st_vec_total += uniskip(cap.view(1, -1)).to(device)
#         # break
#     cap_mean = (cap_st_vec_total / len(
#         captions)).cpu().detach().numpy()  # torch.zeros(1, 2400) #torch.mean(cap_vec, dim=0)
#     np.save(os.path.join(dir, 'coco2014-vgg/coco_caps_st_vec_mean.npy'), cap_mean)

# load novel
with open(dir + 'romance_novel/text.pkl', 'rb') as f:
    novel = pickle.load(f)

novel = [Variable(
    torch.LongTensor([dictionary[word] if word in dictionary.keys() else dictionary['UNK'] for word in l.split()])) for l in novel]
print(len(novel))
# para, para_len = pad(novel, 0)
with torch.no_grad():
    para_st_vec_total = torch.zeros(1, 2400).to(device)
    for para in tqdm(novel):
        para = para.to(device)
        para = uniskip(para.view(1, -1)).to(device)
para_mean = (para_st_vec_total / len(novel)).cpu().detach().numpy()  # torch.zeros(1, 2400) #torch.mean(cap_vec, dim=0)
np.save(os.path.join(dir, 'romance_novel/romantic_st_vec_mean.npy'), para_mean)
