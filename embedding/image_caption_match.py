import os
import pickle as pkl
import json
import numpy
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from textwrap import wrap

import torch

from embedding.dataset import load_dataset
from embedding.model import ImgSenRanking
from embedding.tools import encode_sentences, encode_images

def plot_pictures(images, captions):
    num_lines = len(captions)
    for idx, (image, caption) in enumerate(zip(images, captions)):
        plt.subplot((num_lines - 1) // 5 + 1, 5, idx + 1)
        plt.title(caption, fontsize=10)
        img = Image.open(image)
        plt.imshow(img)
    plt.show()
    plt.clf()

def test_match(saved_info_path,
            model_name,
            param_name,
            data,
            k=10,
            image_path='D:/Dataset/Coco/'):


    # load model options
    param_path = os.path.join(saved_info_path, param_name)
    if not os.path.exists(param_path):
        print(f"{param_path} doesn't exists.")
        return
    with open(param_path, 'rb') as f:
        obj = f.read()
    model_options = {key: val for key, val in pkl.loads(obj, encoding='latin1').items()}
    print(model_options)


    # Load training and development sets
    # train = (train_caps: list [sentence1, sentence2, ...], train_ims: np.array(num_img x 4096) [[feat1], [feat2], ...])
    print('loading dataset')
    train, dev = load_dataset(data)

    # Load dictionary
    worddict = model_options['worddict']
    word_idict = model_options['word_idict']

    # Fixed: caption2image
    caption2img = json.load(open('./caption2image_train.json', 'r'))
    caption2img.update(json.load(open('./caption2image_val.json', 'r')))

    # load model
    model_path = os.path.join(saved_info_path, model_name)
    if not os.path.exists(model_path):
        print(f"{model_path} doesn't exists.")
        return
    img_sen_model = ImgSenRanking(model_options)
    checkpoint = torch.load(model_path)
    img_sen_model.load_state_dict(checkpoint['model_state_dict'])
    img_sen_model = img_sen_model.cuda()
    img_sen_model.eval()

    # encode training captions
    curr_model = {}
    curr_model['options'] = model_options
    curr_model['worddict'] = model_options['worddict']
    curr_model['word_idict'] = model_options['word_idict']
    curr_model['img_sen_model'] = img_sen_model

    all_captions = train[0] + dev[0]

    caption_vec_path = os.path.join(saved_info_path, "coco_caption_vecs.npy")
    if os.path.exists(caption_vec_path):
        caption_vecs = numpy.load(caption_vec_path)
    else:
        caption_vecs = encode_sentences(curr_model, all_captions).cpu().numpy()
        numpy.save(caption_vec_path, caption_vecs)

    img_vecs = encode_images(curr_model, dev[1]).cpu().detach().numpy()

    all_captions = [cap.strip().decode("utf-8") for cap in all_captions]

    n_samples = 0

    for idx, img_vec in enumerate(img_vecs):

        scores = numpy.dot(img_vec, caption_vecs.T).flatten()
        sorted_args = numpy.argsort(scores)[::-1]
        sentences = [all_captions[a][:-2] + '.' if all_captions[a][-1] == '.' and all_captions[a][-2] else all_captions[a] for a in sorted_args[:k]]

        # Update
        x = str(dev[0][idx], 'utf-8')
        if x[-1] == '.' and x[-2] == ' ':
            x = x[:-2] + '.'
        if x not in caption2img:
            continue
        print(f"True caption: {x}")
        print(sentences[:6])
        images, captions = [image_path + caption2img[x]], ['\n'.join(wrap("Ground truth:" + x, 60))]
        for s in sentences:
            if s in caption2img:
                images.append(image_path + caption2img[s])
                captions.append(s)

        if n_samples % 5 == 0:
            plot_pictures(images[:8], captions[:8])

        if (n_samples > 100):
            break

        n_samples += 1

    print('Seen %d samples'%n_samples)

def test_one_pic(saved_info_path,
               model_name,
               param_name,
               test_img_path,
               test_img_feat_path,
               k=100,
               data='coco',
               image_path='D:/Dataset/Coco/'):

    # load model options
    param_path = os.path.join(saved_info_path, param_name)
    if not os.path.exists(param_path):
        print(f"{param_path} doesn't exists.")
        return
    with open(param_path, 'rb') as f:
        obj = f.read()
    model_options = {key: val for key, val in pkl.loads(obj, encoding='latin1').items()}
    print(model_options)

    # Load training and development sets
    # train = (train_caps: list [sentence1, sentence2, ...], train_ims: np.array(num_img x 4096) [[feat1], [feat2], ...])
    print('loading dataset')
    train_caps, dev_caps = load_dataset(data, cap_only=True)

    # Load dictionary
    worddict = model_options['worddict']
    word_idict = model_options['word_idict']

    # Fixed: caption2image
    caption2img = json.load(open('./caption2image_train.json', 'r'))
    caption2img.update(json.load(open('./caption2image_val.json', 'r')))

    # Each sentence in the minibatch have same length (for embedding)
    # Notice: each feature vector has 5 copies
    # train_iter = homogeneous_data.HomogeneousData([train[0], train[1]], batch_size=batch_size, maxlen=maxlen_w)

    model_path = os.path.join(saved_info_path, model_name)
    if not os.path.exists(model_path):
        print(f"{model_path} doesn't exists.")
        return
    img_sen_model = ImgSenRanking(model_options)
    checkpoint = torch.load(model_path)
    img_sen_model.load_state_dict(checkpoint['model_state_dict'])
    img_sen_model = img_sen_model.cuda()
    img_sen_model.eval()

    # encode training captions
    curr_model = {}
    curr_model['options'] = model_options
    curr_model['worddict'] = model_options['worddict']
    curr_model['word_idict'] = model_options['word_idict']
    curr_model['img_sen_model'] = img_sen_model

    all_captions = train_caps + dev_caps

    caption_vec_path = os.path.join(saved_info_path, "coco_caption_vecs.npy")
    if os.path.exists(caption_vec_path):
        caption_vecs = numpy.load(caption_vec_path)
    else:
        caption_vecs = encode_sentences(curr_model, all_captions).cpu().numpy()
        numpy.save(caption_vec_path, caption_vecs)

    if not os.path.exists(test_img_feat_path):
        print(f"{test_img_feat_path} doesn't exist.")
    else:
        with open(test_img_feat_path, 'rb') as f:
            image_feat = numpy.load(f)

    img_vecs = encode_images(curr_model, image_feat).cpu().detach().numpy()

    all_captions = [cap.strip().decode("utf-8") for cap in all_captions]

    for idx, img_vec in enumerate(img_vecs):

        scores = numpy.dot(img_vec, caption_vecs.T).flatten()
        sorted_args = numpy.argsort(scores)[::-1]
        sentences = [
            all_captions[a][:-2] + '.' if all_captions[a][-1] == '.' and all_captions[a][-2] else all_captions[a]
            for a in sorted_args[:k]]

        # Update
        images, captions = [test_img_path], [""]
        for s in sentences:
            if s in caption2img:
                images.append(image_path + caption2img[s])
                captions.append(s)

        plot_pictures(images[:20], captions[:20])

if __name__ == '__main__':
    test_one_pic(saved_info_path='./saved/coco2014-20190317-160243/models',
                 model_name='weights_lstm_batch-34000_score-70.900000.pt',
                 param_name='params_lstm_batch-34000_score-70.900000.pkl',
                 data='coco2014-vgg',
                 test_img_path='../data/test_images/dish1.jpg',
                 test_img_feat_path='../data/test_images/dish1_feat.npy')
    # test_match(saved_info_path='./saved/coco2014-20190317-160243/models',
    #         model_name='weights_lstm_batch-34000_score-70.900000.pt',
    #         param_name='params_lstm_batch-34000_score-70.900000.pkl',
    #         data='coco2014-vgg')