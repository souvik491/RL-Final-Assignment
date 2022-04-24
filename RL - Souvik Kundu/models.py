# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""

import numpy as np
import time

import torch
import torch.nn as nn
from transformer.models import Transformer
import math
import torch.nn.functional as F
from copy import deepcopy
"""
BLSTM (max/mean) encoder
"""
cuda = True
device = torch.device("cuda:0" if cuda else "cpu")
tau = 0.1

class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=False, dropout=self.dpout_model)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        self.enc_lstm.flatten_parameters()
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):

        sent = sent.split() if not tokenize else self.tokenize(sent)
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]

        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing \
                           by "%s %s"..' % (sent, self.bos, self.eos))
        batch = self.get_batch(sent)

        if self.is_cuda():
            batch = batch.cuda()
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs
    
    def getNextHiddenState(self, hc, x):
        hidden = hc[0,0:self.enc_lstm_dim].view(1,1,self.enc_lstm_dim)
        cell = hc[0,self.enc_lstm_dim:].view(1,1,self.enc_lstm_dim)
        input = x.view(1,1,-1) #self.word_embeddings(x).view(1,1,-1)       
        out, hidden = self.enc_lstm(input, [hidden, cell])
        hidden = torch.cat([hidden[0], hidden[1]], -1).view(1, -1)
        return out, hidden


class policyNet(nn.Module):
    def __init__(self, hidden_size, embedding_length):
        super(policyNet, self).__init__()
        self.hidden = 2*hidden_size 
        self.W1 = nn.Linear(self.hidden,1, bias = False)
        self.W2 = nn.Linear(embedding_length,1, bias = False)
        self.W3 = nn.Linear(hidden_size,1, bias = True)
        
        '''
        self.W1 = nn.Parameter(torch.cuda.FloatTensor(2*self.hidden, 1).uniform_(-0.5, 0.5)) 
        self.W2 = nn.Parameter(torch.cuda.FloatTensor(embedding_length, 1).uniform_(-0.5, 0.5)) 
        self.W3 = nn.Parameter(torch.cuda.FloatTensor(embedding_length, 1).uniform_(-0.5, 0.5)) 
        self.b = nn.Parameter(torch.cuda.FloatTensor(1, 1).uniform_(-0.5, 0.5))
        '''

    def forward(self, h, x, hF):
        '''
        h_ = torch.matmul(h.view(1,-1), self.W1) # 1x1
        x_ = torch.matmul(x.view(1,-1), self.W2) # 1x1
        hF_ = torch.matmul(hF.view(1,-1), self.W3) # 1x1
        '''
        h_ = self.W1(h.view(1,-1)) # 1x1
        x_ = self.W2(x.view(1,-1)) # 1x1
        hF_ = self.W3(hF.view(1,-1)) # 1x1
        scaled_out = torch.sigmoid(h_ +  x_ + hF_) # 1x1
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)
        scaled_out = torch.cat([1.0 - scaled_out, scaled_out],0)
        return scaled_out

class actor(nn.Module):
    def __init__(self, hidden_size, embedding_length):
        super(actor, self).__init__()
        self.target_policy = policyNet(hidden_size, embedding_length)
        self.active_policy = policyNet(hidden_size, embedding_length)
     
    def get_target_logOutput(self, h, x):
        out = self.target_policy(h, x)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, h, x, s, scope):
        if scope == "target":
            out = self.target_policy(h, x, s)
        if scope == "active":
            out = self.active_policy(h, x, s)
        return out

    def get_gradient(self, h, x, s, reward, scope):
        if scope == "target":
            out = self.target_policy(h, x, s)
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2
            grad = torch.autograd.grad(logout[index].view(-1), self.target_policy.parameters()) # torch.cuda.FloatTensor(reward[index])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            grad[3].data = grad[3].data * reward[index]
            return grad
        if scope == "active":
            out = self.active_policy(h, x)
        return out

    def assign_active_network_gradients(self, grad1, grad2, grad3, grad4):
        params = [grad1, grad2, grad3, grad4]    
        i=0
        for name, x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i+=1

    def update_target_network(self):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

class critic(nn.Module):
    def __init__(self, config):
        super(critic, self).__init__()
        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.active_pred = eval(self.encoder_type)(config)
        self.target_pred = eval(self.encoder_type)(config)
        
        self.inputdim = 4*1*self.enc_lstm_dim

        if self.nonlinear_fc:
            self.active_classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
            self.target_classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.active_classifier = nn.Sequential(
                nn.Linear(int(self.inputdim), self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )
            self.target_classifier = nn.Sequential(
                nn.Linear(int(self.inputdim), self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )
   
    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.encoder.pos_emb.parameters()))
        dec_freezed_param_ids = set(map(id, self.encoder.decoder.pos_emb.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)
       
    def assign_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1
        params = []
        for name, x in self.active_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_classifier.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def update_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

        params = []
        for name, x in self.active_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_classifier.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1
        params = []
        for name, x in self.target_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_classifier.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def assign_active_network_gradients(self):
        params = []
        #aaa
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i+=1
        for name, x in self.target_pred.named_parameters():
            x.grad = None

        params = []
        for name, x in self.target_classifier.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_classifier.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i+=1
        for name, x in self.target_classifier.named_parameters():
            x.grad = None

    def forward(self, s1, s2, scope):
        # s1 : (s1, s1_len)
        if scope == "target":
            u = self.target_pred(s1)
            v = self.target_pred(s2)
            features = torch.cat((u, v, torch.abs(u-v), u*v), 1)     
            output = self.target_classifier(features)
        
        if scope == "active":
            u = self.active_pred(s1)
            v = self.active_pred(s2)
            features = torch.cat((u, v, torch.abs(u-v), u*v), 1)     
            output = self.active_classifier(features)
        
        return output

    def summary(self, s1):
        emb = self.target_pred(s1)
        return emb
    
    def forward_lstm(self, hc, x, scope):
        if scope == "target":
            out, state = self.target_pred.getNextHiddenState(hc, x)
        if scope == "active":
            out, state = self.active_pred.getNextHiddenState(hc, x)
        return out, state