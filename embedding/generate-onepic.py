import pathlib

import torch
import json
import pickle as pkl
from torch.autograd import Variable
import time
import numpy
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from textwrap import wrap

import embedding.homogeneous_data as homogeneous_data
from embedding.dataset import load_dataset
from embedding.vocab import build_dictionary
from embedding.model import ImgSenRanking, PairwiseRankingLoss
from embedding.tools import encode_sentences, encode_images
from embedding.evaluation import i2t, t2i

def plot_pictures(images, captions):
    num_lines = len(captions)
    for idx, (image, caption) in enumerate(zip(images, captions)):
        plt.subplot((num_lines - 1) // 4 + 1, 4, idx + 1)
        plt.title(caption, fontsize=10)
        img = Image.open(image)
        plt.imshow(img)
    plt.show()
    plt.clf()

def evaluate(model_path, param_path, caption_path,
             test_image,
             dataset_path='D:/Dataset/Coco/',
             data='coco',
             margin=0.2,
             dim=1024,
             dim_image=4096,
             dim_word=300,
             max_epochs=15,
             encoder='lstm',
             dispFreq=10,
             grad_clip=2.0,
             maxlen_w=150,
             batch_size=128,
             validFreq=100,
             early_stop=20,
             lrate=0.01,
             reload_=False):

    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_image'] = dim_image
    model_options['dim_word'] = dim_word
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['batch_size'] = batch_size
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_

    print(model_options)

    # Load training and development sets
    print('loading dataset')
    # train, dev = load_dataset(data)

    with open(test_image, 'rb') as f:
        img_to_test = numpy.load(f)

    with open(param_path, 'rb') as f:
        obj = f.read()
    model_info = {key: val for key, val in pkl.loads(obj, encoding='latin1').items()}

    model_options['worddict'] = model_info['worddict']
    model_options['word_idict'] = model_info['word_idict']
    n_words = len(model_options['worddict'])
    model_options['n_words'] = n_words

    img_sen_model = ImgSenRanking(model_options)

    checkpoint = torch.load(model_path)
    img_sen_model.load_state_dict(checkpoint['model_state_dict'])
    img_sen_model = img_sen_model.cuda()
    img_sen_model.eval()
    uidx = 0
    n_samples = 0

    # Captions
    print('Loading captions...')
    cap = []
    with open(caption_path, 'rb') as f:
        for line in f:
            cap.append(line.strip().decode("utf-8"))
    print('Loading captions done ')

    model_info['img_sen_model'] = img_sen_model

    # caption_vecs = encode_sentences(model_info, cap).cpu().numpy()
    caption_vecs = numpy.load('../saved/new_caption_vecs.npy')
    # with open('../saved/new_caption_vecs.npy', 'wb') as f:
    #     numpy.save(f, caption_vecs)
    # return

    k = 100
    with torch.no_grad():
        # for x, im in zip(dev[0], dev[1]):
        # n_samples += 1
        # uidx += 1

        # Update
        image_vec = encode_images(model_info, img_to_test).cpu().numpy()

        print("Image feature extracted.")
        scores = numpy.dot(image_vec, caption_vecs.T).flatten()
        print("Score!")
        sorted_args = numpy.argsort(scores)[::-1]
        numpy.save("./sorted_score.npy", sorted_args)
        print(sorted_args[:k])
        sentences = [cap[a] for a in sorted_args[:k]]

        for sent in sentences:
            print(sent)

        # x = str(x, 'utf-8')
        # if x not in caption2img:
        #     continue
        # images, captions = [dataset_path+caption2img[x]], ['\n'.join(wrap("Ground truth:"+x,60))]
        # for s in sentences:
        #     if s in caption2img:
        #         images.append(dataset_path+caption2img[s])
        #         captions.append(s)
        #
        # if n_samples % 5 == 0:
        #     plot_pictures(images[:8], captions[:8])
        #
        # print(f"True caption: {x}")
        # print(sentences[:6])
        # if(n_samples > 100):
        #     break
        #
        # print('Seen %d samples'%n_samples)

if __name__ == '__main__':
    evaluate('./saved/coco2014-20190317-160243/models/weights_lstm_batch-20800_score-67.500000.pt',
             './saved/coco2014-20190317-160243/models/params_lstm_batch-20800_score-67.500000.pkl',
            '../data/coco/coco_train_caps.txt',
             test_image='../data/coco/gary_crop_feat.npy')