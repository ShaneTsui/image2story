import pathlib

import torch
import json
import pickle as pkl

import os
import time
import numpy
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from textwrap import wrap

from embedding.dataset import load_dataset
from embedding.model import ImgSenRanking
from embedding.tools import encode_sentences, encode_images

def plot_pictures(images, captions):
    num_lines = len(captions)
    for idx, (image, caption) in enumerate(zip(images, captions)):
        plt.subplot((num_lines - 1) // 4 + 1, 4, idx + 1)
        plt.title(caption, fontsize=10)
        img = Image.open(image)
        plt.imshow(img)
    plt.show()
    plt.clf()

def evaluate(model_path,
             param_path,
             dataset_path='../data/coco/coco2014-vgg19-original',
             image_path='D:/Dataset/Coco/',
             data='coco'):

    # Load dataset
    print('loading dataset')
    train, dev = load_dataset(dataset_path=dataset_path, prefix=data)

    # load parameters
    with open(param_path, 'rb') as f:
        obj = f.read()
    model_info = {key: val for key, val in pkl.loads(obj, encoding='latin1').items()}

    img_sen_model = ImgSenRanking(model_info)
    checkpoint = torch.load(model_path)
    img_sen_model.load_state_dict(checkpoint['model_state_dict'])
    img_sen_model = img_sen_model.cuda()
    img_sen_model.eval()
    model_info['img_sen_model'] = img_sen_model

    # captions
    caption_path = os.path.join(dataset_path, f"{data}_train_caps.txt")
    if not os.path.exists(caption_path):
        print("No train captions")
    print('Loading captions...')

    cap = []
    with open(caption_path, 'rb') as f:
        for line in f:
            cap.append(line.strip().decode("utf-8"))
    print('Loading captions done ')

    caption_vec_path = os.path.join(dataset_path, "coco_caption_vecs.npy")
    if os.path.exists(caption_vec_path):
        caption_vecs = numpy.load(caption_vec_path)
    else:
        caption_vecs = encode_sentences(model_info, cap).cpu().numpy()
        numpy.save(caption_vec_path, caption_vecs)

    # Fixed: caption2image
    caption2img = json.load(open('../saved/caption2image_train.json', 'r'))
    caption2img.update(json.load(open('../saved/caption2image_val.json', 'r')))

    k = 100
    n_samples = 0
    with torch.no_grad():
        for x, im in zip(dev[0], dev[1]):
            n_samples += 1

            # Update
            image_vec = encode_images(model_info, im[None, :]).cpu().numpy()

            scores = numpy.dot(image_vec, caption_vecs.T).flatten()
            sorted_args = numpy.argsort(scores)[::-1]
            sorted_scores = scores[sorted_args]
            sentences = [cap[a] for a in sorted_args[:k]]

            x = str(x, 'utf-8')
            if x[-1] == '.' and x[-2] == ' ':
                x = x[:-2]+'.'
            print(x)
            if x not in caption2img:
                continue
            print(sentences)
            images, captions = [image_path + caption2img[x]], ['\n'.join(wrap("Ground truth:" + x, 60))]
            for s in sentences:
                if s in caption2img:
                    images.append(image_path + caption2img[s])
                    captions.append(s)

            if n_samples % 5 == 0:
                plot_pictures(images[:8], captions[:8])

            print(f"True caption: {x}")
            print(sentences[:6])
            if(n_samples > 100):
                break

            print('Seen %d samples'%n_samples)

if __name__ == '__main__':
    path = '../saved/'
    evaluate(path + 'lstm_batch-23500_score-359.000000.pt',
             path + 'params_lstm_batch-23500_score-359.000000.pkl',
             dataset_path='../data/coco/coco2014-vgg19-original')