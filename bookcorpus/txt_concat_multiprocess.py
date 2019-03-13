import os
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool


old_path = './new'
new_path = './tokenized/'

def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in tqdm(text):
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X


def convert(file):
    with open((os.path.join(old_path, file))) as f:
        with open(os.path.join(new_path, file), 'w') as tf:
            tf.write("\n".join(preprocess(f)))


if not os.path.exists(new_path):
    os.mkdir(new_path)


cpu_num = multiprocessing.cpu_count()
p = Pool(cpu_num)

for root, dirs, files in os.walk(old_path, topdown=False):
    p.map(convert, files)
