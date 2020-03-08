# -*- coding: utf-8 -*- 


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

from jtnn import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os # for save plot

def save_KL_plt(save_dir, epoch, x, kl):
    plt.plot(x, kl)
    plt.xlabel('Iteration')
    plt.ylabel('KL divergence')
    plt.legend(['KL divergence'])
    plt.grid()
    plt.savefig('./plot/{}/KL/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()
def save_Acc_plt(save_dir, epoch, x, word, topo, assm, steo):
    plt.plot(x, word)
    plt.plot(x, topo)
    plt.plot(x, assm)
    plt.plot(x, steo)
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.legend(['Word acc','Topo acc','Assm acc', 'Steo acc'])
    plt.grid()
    plt.savefig('./plot/{}/Acc/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-m", "--model", dest="model_path", default=None)
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-z", "--beta", dest="beta", default=1.0)
parser.add_option("-q", "--lr", dest="lr", default=1e-3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
#beta가 크면 클수록 KL_LOSS가 커진다. 따라서, 더욱 타이트하게 맞추려할 것이다. 논문에서는 0.001 이하를 추천
lr = float(opts.lr)
stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)

if opts.model_path is not None:
    model.load_state_dict(torch.load(opts.model_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

model = model.cuda()
print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

dataset = MoleculeDataset(opts.train_path)

MAX_EPOCH = 7
PRINT_ITER = 20

#print(type(opts))
d=datetime.now()
now = str(d.year)+'_'+str(d.month)+'_'+str(d.day)+'_'+str(d.hour)+'_'+str(d.minute)
folder_name = opts.train_path.split('/')[2] + '_' +now#../data/zinc/train.txt --> zinc
#folder_name = str(datetime.now())
os.makedirs('./plot/'+folder_name+'/KL')        #KL
os.makedirs('./plot/'+folder_name+'/Acc')       #Word, Topo, Assm
print("...Finish Making Plot Folder...")
#Plot
x_plot=[]
kl_plot=[]
word_plot=[]
topo_plot=[]
assm_plot=[]
pnorm_plot=[]
gnorm_plot=[]
steo_plot=[]

for epoch in xrange(MAX_EPOCH):
    start = datetime.now()
    print("EPOCH: %d | TIME: %s " % (epoch+1, str(start)))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x, drop_last=True)

    word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0

    for it, batch in enumerate(dataloader):
        for mol_tree in batch:
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
                    node.cand_mols.append(node.label_mol)

        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta)
            '''
            KL_div: prior distribution p(z)와 Q(z|X,Y)와의 KL div
            Word: Label Prediction acc
            Topo: Topological Prediction acc
            Assm: 조립할 때, 정답과 똑같이 했는가? acc
            dacc(steo): 
            Streo chemical관련, 같은 2D 구조를 가져도 3D 구조로 바꿨을 경우 다른 분자가 나올 수 있으므로 이에 대한 Loss를 따로 둔다. 모델에서는 이 옵션을 쓸 수도 안쓸수도 있음. 
            molecule generation과 따로 분리하여 streo chemical configuration을 다루는 것이 효율적이라고 본문에 언급. 
            논문 [Supplementary Material]에 언급
            '''
            loss.backward()
            optimizer.step()
        except Exception as e:
            print e
            continue

        word_acc += wacc
        topo_acc += tacc
        assm_acc += sacc
        steo_acc += dacc

        if (it + 1) % PRINT_ITER == 0:
            word_acc = word_acc / PRINT_ITER * 100
            topo_acc = topo_acc / PRINT_ITER * 100
            assm_acc = assm_acc / PRINT_ITER * 100
            steo_acc = steo_acc / PRINT_ITER * 100

            print "[%d][%d] KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (epoch, it+1, kl_div, word_acc, topo_acc, assm_acc, steo_acc)
            
            x_plot.append(it)
            kl_plot.append(kl_div)
            word_plot.append(word_acc)
            topo_plot.append(topo_acc)
            assm_plot.append(assm_acc)
            steo_plot.append(steo_acc)
            
            word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
            sys.stdout.flush()

        if (it + 1) % 15000 == 0: #Fast annealing
            scheduler.step()
            print "learning rate: %.6f" % scheduler.get_lr()[0]

        if (it + 1) % 1000 == 0: 
            torch.save(model.state_dict(), opts.save_path + "/model.iter-%d-%d" % (epoch, it + 1))
            
    #Plot per 1 epoch
    print "Cosume Time per Epoch %s" % (str(datetime.now()-start))
    save_KL_plt(folder_name, epoch, x_plot, kl_plot)
    save_Acc_plt(folder_name, epoch, x_plot, word_plot, topo_plot, assm_plot, steo_plot)
    x_plot=[]
    kl_plot=[]
    word_plot=[]
    topo_plot=[]
    assm_plot=[]
    steo_plot=[]

    scheduler.step()
    print "learning rate: %.6f" % scheduler.get_lr()[0]
    torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))

