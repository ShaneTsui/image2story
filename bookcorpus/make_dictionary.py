import numpy
import os
from tqdm import tqdm
import pickle as pkl
from collections import OrderedDict

save_path = './dictionary/'
target_file = "romantic_bbbbnew.txt"
passages = []

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in tqdm(text):
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx + 2   # 0: <eos>, 1: <unk>

    return worddict, wordcount

if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(target_file) as f:
    for line in tqdm(f):
        passages.append(line.strip())

# with open('dictionary_w2i.pkl', 'rb') as f:
#     worddict = pkl.load(f)

worddict, _ = build_dictionary(passages)

with open(os.path.join(save_path, 'romantic_dict.txt'), 'w') as f:
    f.write("\n".join(list(worddict.keys())))

with open(os.path.join(save_path, 'dictionary_w2i.pkl'), 'wb') as f:
    pkl.dump(worddict, f)

word_idict = dict()
for kk, vv in worddict.items():
    word_idict[vv] = kk
word_idict[0] = '<eos>'
word_idict[1] = 'UNK'

with open(os.path.join(save_path, 'dictionary_i2w.pkl'), 'wb') as f:
    pkl.dump(word_idict, f)