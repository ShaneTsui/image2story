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

class GenSample(object):

    def __init__(self, decoder, ctx, k, maxlen, use_unk=True):
        self.decoder = decoder
        self.decoder.eval()
        self.ctx = ctx
        self.k = k
        self.maxlen = maxlen
        self.use_unk = use_unk

    def gen_sample(self):

        sample = []
        sample_score = []

        live_k = 1
        dead_k = 0

        device = torch.device('cuda')
        self.decoder.to(device)

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []

        with torch.no_grad():

            next_w = 0 * numpy.ones((1,)).astype('int64')
            next_w = torch.LongTensor(next_w).view(1, -1).to(device)
            next_state = self.decoder.init_hidden(self.ctx).to(device)

            # next_w: (batch_size, w)
            # next_state: (num_layer=1, batch_size, 2400)
            # output: (batch_size, 1, voc_size=2000)
            for ii in tqdm(range(int(self.maxlen))):
                output, next_state = self.decoder(next_w, next_state)
                next_state = numpy.squeeze(next_state.cpu().detach().numpy(), axis=0)
                next_p = F.softmax(output, dim=2)
                # next_w = torch.multinomial(next_p, 1)
                next_p = next_p.cpu().detach().numpy().reshape(1, -1)

                cand_scores = hyp_scores[:,None] - numpy.log(next_p)
                cand_flat = cand_scores.flatten()

                if not self.use_unk:
                    voc_size = next_p.shape[1]
                    for xx in range(int(len(cand_flat) / voc_size)):
                        cand_flat[voc_size * xx + 1] = 1e20

                ranks_flat = cand_flat.argsort()[:(self.k-dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = numpy.zeros(self.k-dead_k).astype('float32')
                new_hyp_states = []

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[int(ti)]+[int(wi)])
                    new_hyp_scores[idx] = copy.copy(costs[int(idx)])
                    new_hyp_states.append(copy.copy(next_state[int(ti)]))
                sample, sample_score, hyp_samples, hyp_scores, hyp_states, new_live_k, dead_k = self.__check_finish_samples( sample, sample_score,
                new_hyp_samples, new_hyp_scores, new_hyp_states, dead_k)

                hyp_scores = numpy.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= self.k:
                    break

                next_w = torch.LongTensor(numpy.array([[w[-1]] for w in hyp_samples])).to(device)
                next_state = torch.squeeze(torch.from_numpy(numpy.array(hyp_states)), dim=1).unsqueeze(0).to(device)

            if live_k > 0:
                for idx in range(int(live_k)):
                    idx=int(idx)
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

            return sample, sample_score

    def generate_story(self, sample, dictionary):
        # 1: <eos>, 2: <unk>, 0: end_padding
        word_idict = dict()
        for kk, vv in dictionary.items():
            word_idict[vv] = kk
        neuralstory = " ".join([word_idict[id] for id in sample])
        return neuralstory


    def __check_finish_samples(self, sample, sample_score, new_hyp_samples, new_hyp_scores, new_hyp_states, dead_k):
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []

        for idx in range(len(new_hyp_samples)):
            idx = int(idx)
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])

        return sample, sample_score, hyp_samples, hyp_scores, hyp_states, new_live_k, dead_k


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
    # sentence = ['I', 'like', 'my', 'teacher', 'and', 'his', 'dog', '.', '<eos>']
    sentence = ['a', 'man', 'and', 'a', 'woman', 'dressed', 'in', 'wedding', 'outfits', 'with', 'a', 'microphone', '.', '<eos>']
    sentence_idx = Variable(
        torch.LongTensor([dictionary[w] if w in dictionary else dictionary['UNK'] for w in sentence]))
    print(sentence_idx)
    ctx = uniskip(sentence_idx.view(1, -1)).to(device)

    generator = GenSample(decoder, ctx.unsqueeze(0), k=10, maxlen=100, use_unk=False)
    samples, sample_scores = generator.gen_sample()
    print(samples)
    for sample, sample_score in zip(samples, sample_scores):
        story = generator.generate_story(sample, dictionary)
        print(story, sample_score)