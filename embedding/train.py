import pathlib

import torch
import os
import pickle as pkl
from torch.autograd import Variable
import time
import numpy
import tqdm

import embedding.homogeneous_data as homogeneous_data
from embedding.dataset import load_dataset
from embedding.vocab import build_dictionary
from embedding.model import ImgSenRanking, PairwiseRankingLoss
from embedding.tools import encode_sentences, encode_images
from embedding.evaluation import i2t, t2i

def trainer(data='coco',
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
            # saveto='../saved/vse/coco/', # Should be a folder path, so add a "/" at the end
            validFreq=100,
            early_stop=20,
            lrate=0.0002,
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
    # model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_

    print(model_options)

    # Set up folder for model saving
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    saved_model_path = '{}/saved/{}-{}/models/'.format(os.getcwd(), data, cur_time)
    model_pathlib = pathlib.Path(saved_model_path)
    if not model_pathlib.exists():
        pathlib.Path(model_pathlib).mkdir(parents=True, exist_ok=True)

    # Set up folder for model saving
    saved_result_path = '{}/saved/{}-{}/results/'.format(os.getcwd(), data, cur_time)
    result_pathlib = pathlib.Path(saved_result_path)
    if not result_pathlib.exists():
        pathlib.Path(result_pathlib).mkdir(parents=True, exist_ok=True)


    # reload options
    if reload_ and os.path.exists(saved_model_path):
        print('reloading...' + saved_model_path)
        with open('%s.pkl' % saved_model_path, 'rb') as f:
            model_options = pkl.load(f)


    # Load training and development sets
    # train = (train_caps: list [sentence1, sentence2, ...], train_ims: np.array(num_img x 4096) [[feat1], [feat2], ...])
    print('loading dataset')
    train, dev = load_dataset(data)


    # Create and save dictionary
    # worddict: worddict['<eos>'] = 0
    print('Create dictionary')
    worddict, _ = build_dictionary(train[0] + dev[0])
    n_words = len(worddict)
    model_options['n_words'] = n_words
    print('Dictionary size: ' + str(n_words))
    # Save dictionary
    with open('%sdictionary.pkl' % saved_model_path, 'wb') as f:
        pkl.dump(worddict, f)


    # Inverse dictionary
    # word_idict[0] = '<eos>'
    word_idict = dict()
    for kk, vv in worddict.items():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    model_options['worddict'] = worddict
    model_options['word_idict'] = word_idict


    # Each sentence in the minibatch have same length (for embedding)
    # Notice: each feature vector has 5 copies
    train_iter = homogeneous_data.HomogeneousData([train[0], train[1]], batch_size=batch_size, maxlen=maxlen_w)

    img_sen_model = ImgSenRanking(model_options)
    img_sen_model = img_sen_model.cuda()

    loss_fn = PairwiseRankingLoss(margin=margin)
    loss_fn = loss_fn.cuda()

    params = filter(lambda p: p.requires_grad, img_sen_model.parameters())
    optimizer = torch.optim.Adam(params, lrate)

    uidx = 0
    curr = 0.0
    n_samples = 0

    # For Early-stopping
    best_r1, best_r5, best_r10, best_medr = 0.0, 0.0, 0.0, 0
    best_r1i, best_r5i, best_r10i, best_medri = 0.0, 0.0, 0.0, 0
    best_step = 0

    for eidx in range(max_epochs):

        print('Epoch ', eidx)

        for x, im in train_iter:
            n_samples += len(x)
            uidx += 1

            x, im = homogeneous_data.prepare_data(x, im, worddict, maxlen=maxlen_w, n_words=n_words)

            if x is None:
                print('Minibatch with zero sample under length ', maxlen_w)
                uidx -= 1
                continue

            x = Variable(torch.from_numpy(x).cuda())
            im = Variable(torch.from_numpy(im).cuda())
            # Update
            x, im = img_sen_model(x, im)
            cost = loss_fn(im, x)
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm(params, grad_clip)
            optimizer.step()

            if numpy.mod(uidx, dispFreq) == 0:
                print('Epoch ', eidx, '\tUpdate ', uidx, '\tCost ', cost.data.item())

            if numpy.mod(uidx, validFreq) == 0:

                print('Computing results...')
                curr_model = {}
                curr_model['options'] = model_options
                curr_model['worddict'] = worddict
                curr_model['word_idict'] = word_idict
                curr_model['img_sen_model'] = img_sen_model

                ls, lim = encode_sentences(curr_model, dev[0]), encode_images(curr_model, dev[1])

                r_time = time.time()
                (r1, r5, r10, medr) = i2t(lim, ls)
                print("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
                (r1i, r5i, r10i, medri) = t2i(lim, ls)
                print("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))

                # print("Cal Recall@K using %ss" %(time.time()-r_time))

                curr_step = uidx / validFreq
                currscore = r1 + r5 + r10 + r1i + r5i + r10i

                print(f"cur score = {currscore}")
                if currscore > curr:
                    curr = currscore
                    best_r1, best_r5, best_r10, best_medr = r1, r5, r10, medr
                    best_r1i, best_r5i, best_r10i, best_medri = r1i, r5i, r10i, medri
                    best_step = curr_step

                    # Save model
                    print(f'Saving model with score {currscore}',)
                    pkl.dump(model_options, open('%sparams_%s_batch-%s_score-%f.pkl'%(saved_model_path, encoder, uidx, currscore), 'wb'))
                    torch.save({
                        'model_state_dict': img_sen_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, '%sweights_%s_batch-%s_score-%f.pt'%(saved_model_path, encoder, uidx, currscore))
                    # torch.save(img_sen_model.state_dict(), '%s_model_%s.pkl'%(saved_model_path, embedding))
                    print('Done')

                if curr_step - best_step > early_stop:
                    print('Early stopping ...')
                    print("Image to text: %.1f, %.1f, %.1f, %.1f" % (best_r1, best_r5, best_r10, best_medr))
                    print("Text to image: %.1f, %.1f, %.1f, %.1f" % (best_r1i, best_r5i, best_r10i, best_medri))
                    return 0


        print('Seen %d samples'%n_samples)

if __name__ == '__main__':
    trainer()