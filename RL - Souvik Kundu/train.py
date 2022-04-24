import os
import sys
import time
import argparse
from collections import Counter
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import critic, actor
from read_data import get_SICK_data
print("sick: model_actor_avg0.25_17")
parser = argparse.ArgumentParser(description='NLI training')

samplecnt = 1
epsilon = 0.05
alpha = 0.1
verbose = 1
if verbose:
    f = open("results.txt", "w")
    f.close()

bothCritic = "both_critic0.25abs_nodelay.pickle-ds"
bothActor = "both_actor0.25abs_nodelay.pickle-ds"


# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/SOUVIK/', help="Output directory")
parser.add_argument("--criticmodelname", type=str, default='model_best_souvik.pickle')
parser.add_argument("--actormodelname", type=str, default='model_actor_avg0.25.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=1, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.01", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=1, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=1024, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
#train, valid, test = get_nli(params.nlipath)

train, valid, test = get_SICK_data()

word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)


for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([[word for word in sent.split() if word in word_vec] for sent in eval(data_type)[split]])



"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = critic(config_nli_model)
actorModel = actor(params.enc_lstm_dim, params.word_emb_dim)
print(nli_net)
print(actorModel)


for name, x in nli_net.named_parameters():
    print(name)

for name, x in actorModel.named_parameters():
    print(name)

#print(nli_net.target_pred.enc_lstm.weight_ih_l0)
#print(nli_net.target_classifier[4].bias)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False


# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
critic_target_optimizer = optim_fn(list(nli_net.target_pred.parameters()) + list(nli_net.target_classifier.parameters()), **optim_params)

optim_fn2, optim_params2 = get_optimizer(params.optimizer)
critic_active_optimizer = optim_fn(list(nli_net.active_pred.parameters()) + list(nli_net.active_classifier.parameters()), **optim_params2)


optim_fn3, optim_params3 = get_optimizer("adam,lr=0.1")
actor_target_optimizer = optim_fn3(actorModel.target_policy.parameters(), **optim_params3)

optim_fn4, optim_params4 = get_optimizer("adam,lr=0.1")
actor_active_optimizer = optim_fn4(actorModel.active_policy.parameters(), **optim_params4)

# cuda by default
nli_net.cuda()
actorModel.cuda()
loss_fn.cuda()


def Sampling_RL(current, summary, length, epsilon, Random = True):
    current_lower_state = torch.zeros(1, 2*params.enc_lstm_dim).cuda()
    current = current.squeeze(0)
    actions = []
    states = []
    for pos in range(0, length):
        predicted = actorModel.get_target_output(current_lower_state, current[pos], summary, scope = "target")
        states.append([current_lower_state, current[pos], summary])
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < float(predicted[0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[0].item()) else 0)
        else:
            action = int(torch.argmax(predicted))
        actions.append(action)
        if action == 1:
            out_d, current_lower_state = nli_net.forward_lstm(current_lower_state, current[pos], scope = "target")

    Rinput = []
    for (i, a) in enumerate(actions):
        if a == 1:
            Rinput.append(current[i])
    Rlength = len(Rinput)
    
    if Rlength == 0:
        actions[length-2] = 1
        Rinput.append(current[length-2])
        Rlength = 1
    
    Rinput = torch.stack(Rinput)
    return actions, states, Rinput, Rlength
    

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params2['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch, RL_train = True, LSTM_train = True):
    print('\nTRAINING : Epoch ' + str(epoch))
    
    actorModel.train(False)
    nli_net.train(False)
    if RL_train:
        print("Actor Training")
        print('Learning rate : {0}'.format(actor_active_optimizer.param_groups[0]['lr']))
        actorModel.train()
    if LSTM_train:
        print("InferSent Training")
        critic_active_optimizer.param_groups[0]['lr'] = critic_active_optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else critic_active_optimizer.param_groups[0]['lr']
        print('Learning rate : {0}'.format(critic_active_optimizer.param_groups[0]['lr']))
        nli_net.train()
    
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    for stidx in tqdm(range(0, len(s1), params.batch_size)):
        
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + s1_batch.size(1)])).cuda()
        k = s1_batch.size(1)  # actual batch size
        predict = torch.zeros(s1_batch.size(1), params.n_classes).cuda()
        avgloss = 0.
        totloss = 0.
        nli_net.assign_active_network()
        actorModel.assign_active_network()
        #print("Target Weight: ", actorModel.target_policy.W1.weight.data, "\n\n")
        for kk in range(s1_batch.size(1)):
            left = s1_batch.transpose(0,1)[kk].view(-1, 1, 300)
            right = s2_batch.transpose(0,1)[kk].view(-1, 1, 300)
            left_len = np.array([s1_len[kk]])
            right_len = np.array([s2_len[kk]])
            tgt = tgt_batch[kk].view(-1)
            if RL_train:
                leftSummary = nli_net.summary((left, left_len))[-1]
                rightSummary = nli_net.summary((right, right_len))[-1]
                actionlist_left, actionlist_right, statelist_left, statelist_right, losslist = [], [], [], [], []
                aveloss = 0.
                for i in range(samplecnt):
                    actions_left, states_left, Rinput_left, Rlength_left = Sampling_RL(left, rightSummary, int(left_len), epsilon, Random=True)
                    actions_right, states_right, Rinput_right, Rlength_right = Sampling_RL(right, leftSummary, int(right_len), epsilon, Random=True)
                    actionlist_left.append(actions_left)
                    statelist_left.append(states_left)
                    actionlist_right.append(actions_right)
                    statelist_right.append(states_right)
                    out = nli_net((Rinput_left, np.array([Rlength_left])), (Rinput_right, np.array([Rlength_right])), scope = "target")
                    loss_ = loss_fn(out, tgt)
                    lossL = (((float(Rlength_left) /  int(left.size(1))) + (int(left.size(1)) / float(Rlength_left)) * 0.25) - 1.0)
                    lossR = (((float(Rlength_right) /  int(right.size(1))) + (int(right.size(1)) / float(Rlength_right)) * 0.25) - 1.0)
                    loss_ =  loss_ + ((lossL + lossR)/2) * 0.1 * params.n_classes
                    aveloss += loss_
                    losslist.append(loss_)
                aveloss /= samplecnt
                totloss += aveloss
                grad1 = None
                grad2 = None
                grad3 = None
                grad4 = None
                flag = 0 
                if LSTM_train:
                    critic_active_optimizer.zero_grad()
                    critic_target_optimizer.zero_grad()
                    actions_left, states_left, Rinput_left, Rlength_left = Sampling_RL(left, rightSummary, int(left_len), epsilon, Random=False)
                    actions_right, states_right, Rinput_right, Rlength_right = Sampling_RL(right, leftSummary, int(right_len), epsilon, Random=False)
                    output = nli_net((Rinput_left, np.array([Rlength_left])), (Rinput_right, np.array([Rlength_right])), scope = "target")
                    predict[kk] = output
                    loss = loss_fn(output, tgt)
                    avgloss += loss.item()
                    loss.backward()
                    nli_net.assign_active_network_gradients()
                    shrink_factor = 1
                    total_norm = 0
                    for p in nli_net.active_pred.parameters():
                        if p.requires_grad:
                            p.grad.data.div_(k ** 2)  # divide by the actual batch size
                            total_norm += p.grad.data.norm() ** 2
                    for p in nli_net.active_classifier.parameters():
                        if p.requires_grad:
                            p.grad.data.div_(k ** 2)  # divide by the actual batch size
                            total_norm += p.grad.data.norm() ** 2
                    total_norm = np.sqrt(total_norm.cpu())
                    if total_norm > params.max_norm:
                        shrink_factor = params.max_norm / total_norm
                    current_lr = critic_active_optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
                    critic_active_optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
                    critic_active_optimizer.param_groups[0]['lr'] = current_lr            
                    critic_active_optimizer.step()
                actor_target_optimizer.zero_grad()
                for i in range(samplecnt): #5
                    for pos in range(len(actionlist_left[i])): #19 --> 13
                        rr = [0, 0]
                        rr[actionlist_left[i][pos]] = ((losslist[i] - aveloss) * alpha).cpu().item()
                        g = actorModel.get_gradient(statelist_left[i][pos][0], statelist_left[i][pos][1], statelist_left[i][pos][2], rr, scope = "target")
                        if flag == 0:
                            grad1 = g[0]
                            grad2 = g[1]
                            grad3 = g[2]
                            grad4 = g[3]
                            flag = 1
                        else:
                            grad1 += g[0]
                            grad2 += g[1]
                            grad3 += g[2]
                            grad4 += g[3]
                    for pos in range(len(actionlist_right[i])): # 25 --> 5
                        rr = [0, 0]
                        rr[actionlist_right[i][pos]] = ((losslist[i] - aveloss) * alpha).cpu().item()
                        g = actorModel.get_gradient(statelist_right[i][pos][0], statelist_right[i][pos][1], statelist_right[i][pos][2], rr, scope = "target")
                        grad1 += g[0]
                        grad2 += g[1]
                        grad3 += g[2]
                        grad4 += g[3]
                actor_active_optimizer.zero_grad()
                actorModel.assign_active_network_gradients(grad1, grad2, grad3, grad4)
                
                actor_active_optimizer.step()
                #output = nli_net((left, left_len), (right, right_len), "target")
                _, _, Rinput_left, Rlength_left = Sampling_RL(left, rightSummary, int(left_len), epsilon, Random=False)
                _, _, Rinput_right, Rlength_right = Sampling_RL(right, leftSummary, int(right_len), epsilon, Random=False)
                output = nli_net((Rinput_left, np.array([Rlength_left])), (Rinput_right, np.array([Rlength_right])), scope = "target")
                predict[kk] = output
            else:
                critic_active_optimizer.zero_grad()
                critic_target_optimizer.zero_grad()
                output = nli_net((left, left_len), (right, right_len), "target")
                predict[kk] = output
                loss = loss_fn(output, tgt)
                avgloss += loss.item()
                loss.backward()
                nli_net.assign_active_network_gradients()
                shrink_factor = 1
                total_norm = 0
                for p in nli_net.active_pred.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(k ** 2)  # divide by the actual batch size
                        total_norm += p.grad.data.norm() ** 2
                for p in nli_net.active_classifier.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(k ** 2)  # divide by the actual batch size
                        total_norm += p.grad.data.norm() ** 2
                total_norm = np.sqrt(total_norm.cpu())
                if total_norm > params.max_norm:
                    shrink_factor = params.max_norm / total_norm
                current_lr = critic_active_optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
                critic_active_optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
                critic_active_optimizer.param_groups[0]['lr'] = current_lr            
                critic_active_optimizer.step()
        if RL_train:
            pass
            #actorModel.update_target_network()
            '''
            pred = predict.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
            assert len(pred) == len(s1[stidx:stidx + params.batch_size])

            # loss
            all_costs.append(avgloss/params.batch_size)
            words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

            #print(nli_net.classifier[4].bias)

            if len(all_costs) == 100:
                logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                                stidx, round(np.mean(all_costs), 2),
                                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                                int(words_count * 1.0 / (time.time() - last_time)),
                                round(100.*correct.item()/(stidx+k), 2)))
                print(logs[-1])
                last_time = time.time()
                words_count = 0
                all_costs = []
            '''
           
            if LSTM_train:
                nli_net.update_target_network()
                pred = predict.data.max(1)[1]
                correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
                assert len(pred) == len(s1[stidx:stidx + params.batch_size])

                # loss
                all_costs.append(avgloss/params.batch_size)
                words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

                #print(nli_net.classifier[4].bias)

                if len(all_costs) == 100:
                    logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                                    stidx, round(np.mean(all_costs), 2),
                                    int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                                    int(words_count * 1.0 / (time.time() - last_time)),
                                    round(100.*correct.item()/(stidx+k), 2)))
                    print(logs[-1])
                    last_time = time.time()
                    words_count = 0
                    all_costs = []
        else:
            nli_net.assign_target_network()
            pred = predict.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
            assert len(pred) == len(s1[stidx:stidx + params.batch_size])

            # loss
            all_costs.append(avgloss/params.batch_size)
            words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

            

            if len(all_costs) == 100:
                logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                                stidx, round(np.mean(all_costs), 2),
                                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                                int(words_count * 1.0 / (time.time() - last_time)),
                                round(100.*correct.item()/(stidx+k), 2)))
                print(logs[-1])
                last_time = time.time()
                words_count = 0
                all_costs = []
    if LSTM_train:
        train_acc = round(100 * correct.item()/len(s1), 2)
        print('results : epoch {0} ; mean accuracy train : {1}'.format(epoch, train_acc))
        return train_acc
    else:
        return None

def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    if eval_type == "train":
        s1  = train['s1']
        s2  = train['s2']
        target  = train['label']
    if eval_type == "test":
        s1  = test['s1']
        s2  = test['s2']
        target  = test['label']
    if eval_type == "valid":
        s1  = valid['s1']
        s2  = valid['s2']
        target  = valid['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len), "target")

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.criticmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                critic_active_optimizer.param_groups[0]['lr'] = critic_active_optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              critic_active_optimizer.param_groups[0]['lr']))
                if critic_active_optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc

def evaluate_RL(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    actorModel.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    if eval_type == "train":
        s1  = train['s1']
        s2  = train['s2']
        target  = train['label']
    if eval_type == "test":
        s1  = test['s1']
        s2  = test['s2']
        target  = test['label']
    if eval_type == "valid":
        s1  = valid['s1']
        s2  = valid['s2']
        target  = valid['label']

    ll, rl, ll_, rl_ = 0, 0, 0, 0
    deleteCount = dict()
    wordCount = dict()
    for i in range(0, len(s1)):
        if i % 100 == 0:
            print("Evaluating... ", i)
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + 1], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + 1], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + 1])).cuda()

        # model forward
        leftSummary = nli_net.summary((s1_batch, s1_len))[-1]
        rightSummary = nli_net.summary((s2_batch, s2_len))[-1]
        actions_left, states_left, Rinput_left, Rlength_left = Sampling_RL(s1_batch, rightSummary, int(s1_len), epsilon, Random=False)
        actions_right, states_right, Rinput_right, Rlength_right = Sampling_RL(s2_batch, leftSummary, int(s2_len), epsilon, Random=False)
        #print(s1_batch.size(), actions_left, Rinput_left.size(), s2_batch.size(), actions_right, Rinput_right.size(), "\n\n")
        output = nli_net((Rinput_left, np.array([Rlength_left])), (Rinput_right, np.array([Rlength_right])), scope = "target")

        pred = output.data.max(1)[1]

        if verbose:
            sourceL = s1[i:i + 1][0]
            sourceR = s2[i:i + 1][0]
            tempL, tempR = [], []
            
            for x in range(1,len(actions_left)-1):
                
                if sourceL[x] not in wordCount.keys():
                    wordCount[sourceL[x]] = 1
                else:
                    wordCount[sourceL[x]] += 1
                
                if actions_left[x] == 1:
                    tempL.append(sourceL[x])
                if actions_left[x] == 0:
                    if sourceL[x] not in deleteCount.keys():
                        deleteCount[sourceL[x]] = 1
                    else:
                        deleteCount[sourceL[x]] += 1
            
            for x in range(1,len(actions_right)-1):
                
                if sourceR[x] not in wordCount.keys():
                    wordCount[sourceR[x]] = 1
                else:
                    wordCount[sourceR[x]] += 1

                if actions_right[x] == 1:
                    tempR.append(sourceR[x])
                if actions_right[x] == 0:
                    if sourceR[x] not in deleteCount.keys():
                        deleteCount[sourceR[x]] = 1
                    else:
                        deleteCount[sourceR[x]] += 1
            
            with open("results.txt", "a") as f:
                f.write(" ".join(sourceL[1:-1]) + "-----" + " ".join(sourceR[1:-1]) + "\n")
                f.write(" ".join(tempL) + "-----" + " ".join(tempR) + "\nactual: " + str(int(tgt_batch)) + " pred: " + str(int(pred)) + "\n\n")
            ll += len(actions_left)
            rl += len(actions_right)
            ll_ += Counter(actions_left)[1]
            rl_ += Counter(actions_right)[1]            
        
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
    with open("results.txt", "a") as f:
        f.write("Average left: " + str(ll/len(s1)) + "\nAverage left new: " + str(ll_/len(s1)) + "\nAverage right: " + str(rl/len(s1)) + "\nAverage right new: " + str(rl_/len(s1)))
        #f.write(deleteCount + "\n\n\n" + wordCount)
        for key, value in sorted(deleteCount.items(), key=lambda item: item[1]):
            f.write(str(key) + ":" + str(value) + "\n")
        f.write("\n\n\n")
        for key, value in sorted(wordCount.items(), key=lambda item: item[1]):
            f.write(str(key) + ":" + str(value) + "\n")
    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    print(eval_type, " accuracy: ", eval_acc)
    
    if final_eval:
        params.criticmodelname = bothCritic
        params.actormodelname = bothActor
        
    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(actorModel.state_dict(), os.path.join(params.outputdir, params.actormodelname))
            if final_eval:
                torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.criticmodelname))
            val_acc_best = eval_acc
        else:
            if final_eval:
                if 'sgd' in params.optimizer:
                    critic_active_optimizer.param_groups[0]['lr'] = critic_active_optimizer.param_groups[0]['lr'] / params.lrshrink
                    print('Shrinking lr by : {0}. New lr = {1}'
                          .format(params.lrshrink,
                                  critic_active_optimizer.param_groups[0]['lr']))
                    if critic_active_optimizer.param_groups[0]['lr'] < params.minlr:
                        stop_training = True
                if 'adam' in params.optimizer:
                    # early stopping (at 2nd decrease in accuracy)
                    stop_training = adam_stop
                    adam_stop = True
    return eval_acc




''' INITIAL CRITIC TRAIN
"""
Train model on Natural Language Inference task
"""
epoch = 1
while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch, RL_train = False)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
print("\nCritic Loaded")
print(evaluate(epoch, 'test'))
print(evaluate(epoch, 'valid'))
'''

'''
print("ACTOR TRAIN")
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
epoch = 1
while not stop_training and epoch <= params.n_epochs:
    print(trainepoch(epoch, LSTM_train = False))
    eval_acc = evaluate_RL(epoch, 'valid')
    epoch += 1

nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
print("\nCritic Loaded")
actorModel.load_state_dict(torch.load(os.path.join(params.outputdir, params.actormodelname)))
print("\nActor Loaded")
#print(evaluate_RL(epoch, 'train'))
#print(evaluate_RL(epoch, 'test'))
print(evaluate_RL(epoch, 'test'))
'''

print("FINAL CRITIC  TRAIN")
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.criticmodelname)))
print("\nCritic Loaded")
actorModel.load_state_dict(torch.load(os.path.join(params.outputdir, params.actormodelname)))
print("\nActor Loaded")
epoch = 1
while not stop_training and epoch <=params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate_RL(epoch, 'valid', final_eval = True)
    print(eval_acc)
    epoch += 1
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, bothCritic)))
print("\nCritic Loaded")
actorModel.load_state_dict(torch.load(os.path.join(params.outputdir, bothActor)))
print("\nActor Loaded")
print(evaluate_RL(epoch, 'test'))


