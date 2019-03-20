"""
Code for sequence generation
"""
import os
import numpy
import copy
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from decoder.talker import Decoder
from skipthoughts import UniSkip
import decoder.config as decoder_config

def generate_story(sample, dictionary):
    # 1: <eos>, 2: <unk>, 0: end_padding
    word_idict = dict()
    for kk, vv in dictionary.items():
        word_idict[vv] = kk
    neuralstory = " ".join([word_idict[id] for id in sample])
    return neuralstory

def gen_sample(decoder, ctx, k=1, maxlen=100,
               stochastic=False, argmax=False, use_unk=True):
    """
    Generate a sample, using either beam search or stochastic sampling
    """
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    device = torch.device('cuda')
    decoder.to(device)

    with torch.no_grad():
        next_w = 0 * numpy.ones((1,)).astype('int64')
        next_w = torch.LongTensor(next_w).view(1, -1).to(device)
        next_state = decoder.init_hidden(ctx).to(device)

        for ii in tqdm(range(maxlen)):

            output, next_state = decoder(next_w, next_state)
            next_state = numpy.squeeze(next_state.cpu().detach().numpy(), axis=0)
            next_p = F.softmax(output, dim=2)
            next_p = next_p.cpu().detach().numpy().reshape(1, -1)

            if stochastic:
                if argmax:
                    nw = next_p[0].argmax()
                else:
                    nw = next_w[0]
                sample.append(nw)
                sample_score += next_p[0,nw]
                if nw == 0:
                    break
            else:
                cand_scores = hyp_scores[:,None] - numpy.log(next_p)
                cand_flat = cand_scores.flatten()

                if not use_unk:
                    voc_size = next_p.shape[1]
                    for xx in range(len(cand_flat) / voc_size):
                        cand_flat[voc_size * xx + 1] = 1e20

                ranks_flat = cand_flat.argsort()[:(k-dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat // voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
                new_hyp_states = []

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []

                for idx in range(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                hyp_scores = numpy.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = torch.LongTensor(numpy.array([[w[-1]] for w in hyp_samples])).to(device)
                next_state = torch.squeeze(torch.from_numpy(numpy.array(hyp_states)), dim=1).unsqueeze(0).to(device)

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in range(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        return sample, sample_score

if __name__=="__main__":
    cwd = os.getcwd()
    # with open('config.yaml') as f:
    #     config = yaml.load(f)

    skip_dir = '../data/skip-thoughts'

    # load dictionary
    print('Loading dictionary...')
    with open(decoder_config.paths['dictionary'], 'rb') as f:
        dictionary = pickle.load(f)
    dict_size = decoder_config.settings['decoder']['n_words']
    print('Dictionary successfully loaded.\n' + '*' * 50)

    # load skipvector
    print('Loading skip-thoughts...')

    uniskip = UniSkip(skip_dir, list(dictionary.keys()))
    print('Skip-thoughts successfully loaded.\n' + '*' * 50)

    # check GPU
    assert torch.cuda.is_available(), 'CUDA is not available'
    device = torch.device('cuda')

    # load model
    decoder = Decoder(vocab_size=decoder_config.settings['decoder']['n_words'], \
                    embed_size=decoder_config.settings['decoder']['dim_word'], \
                    hidden_size=decoder_config.settings['decoder']['dim']).to(device)
    decoder.load_state_dict(torch.load(decoder_config.settings['decoder']['model_path']))
    decoder.eval()

    # Generate st-vec encoding
    sentence = ['I', 'like', 'my', 'teacher', 'and', 'his', 'dog', '.', '<eos>']
    sentence_idx = Variable(
        torch.LongTensor([dictionary[w] if w in dictionary else dictionary['UNK'] for w in sentence]))
    print(sentence_idx)
    ctx = uniskip(sentence_idx.view(1, -1)).to(device)

    samples, sample_scores = gen_sample(decoder, ctx.unsqueeze(0), k=1, stochastic=False, maxlen=100, use_unk=True)
    print(samples, sample_scores)
    # for sample, sample_score in zip(samples, sample_scores):
    #     story = generate_story(sample, dictionary)
    #     print(story, sample_score)
