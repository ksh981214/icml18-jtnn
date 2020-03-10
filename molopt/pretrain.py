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
    plt.savefig('./pretrain_plot/{}/KL/epoch_{}.png'.format(str(save_dir),str(epoch)))
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
    plt.savefig('./pretrain_plot/{}/Acc/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()
    
def save_Prop_plt(save_dir, epoch, x, prop):
    plt.plot(x, prop)
    plt.xlabel('Iteration')
    plt.ylabel('Property(y(m))')
    plt.grid()
    plt.savefig('./pretrain_plot/{}/Prop/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-p", "--prop", dest="prop_path")
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

model = JTPropVAE(vocab, hidden_size, latent_size, depth)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant(param, 0)
    else:
        nn.init.xavier_normal(param)

model = model.cuda()
print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

dataset = PropDataset(opts.train_path, opts.prop_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x)

MAX_EPOCH = 3
PRINT_ITER = 20

d=datetime.now()
now = str(d.year)+'_'+str(d.month)+'_'+str(d.day)+'_'+str(d.hour)+'_'+str(d.minute)
folder_name = opts.train_path.split('/')[2] + '_' +now#../data/zinc/train.txt --> zinc
os.makedirs('./pretrain_plot/'+folder_name+'/KL')        #KL
os.makedirs('./pretrain_plot/'+folder_name+'/Acc')       #Word, Topo, Assm
os.makedirs('./pretrain_plot/'+folder_name+'/Prop')       #Prop

print("...Finish Making Plot Folder...")
#Plot
x_plot=[]
kl_plot=[]

word_plot=[]
topo_plot=[]
assm_plot=[]
steo_plot=[]
prop_plot=[]

for epoch in xrange(MAX_EPOCH):
    start = datetime.now()
    print("EPOCH: %d | TIME: %s " % (epoch+1, str(start)))
    
    
    word_acc,topo_acc,assm_acc,steo_acc,prop_acc = 0,0,0,0,0

    for it, batch in enumerate(dataloader):
        for mol_tree,_ in batch:
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
                    node.cand_mols.append(node.label_mol)

        model.zero_grad()
        loss, kl_div, wacc, tacc, sacc, dacc, pacc = model(batch, beta=0)
        '''
            KL_div: prior distribution p(z)와 Q(z|X,Y)와의 KL div
            Word: Label Prediction acc
            Topo: Topological Prediction acc
            Assm: 조립할 때, 정답과 똑같이 했는가? acc
            dacc(steo): 
            Streo chemical관련, 같은 2D 구조를 가져도 3D 구조로 바꿨을 경우 다른 분자가 나올 수 있으므로 이에 대한 Loss를 따로 둔다. 모델에서는 이 옵션을 쓸 수도 안쓸수도 있음. 
            molecule generation과 따로 분리하여 streo chemical configuration을 다루는 것이 효율적이라고 본문에 언급. 
            논문 [Supplementary Material]에 언급
            pacc(Property): property predictor F to predict y(m) = logP(m) - SA(m)
            여기선 acc로 받지만, 모델에서 보내주는건 사실 MSEloss임.
        '''
        
        
        loss.backward()
        optimizer.step()

        word_acc += wacc
        topo_acc += tacc
        assm_acc += sacc
        steo_acc += dacc
        prop_acc += pacc

        if (it + 1) % PRINT_ITER == 0:
            word_acc = word_acc / PRINT_ITER * 100
            topo_acc = topo_acc / PRINT_ITER * 100
            assm_acc = assm_acc / PRINT_ITER * 100
            steo_acc = steo_acc / PRINT_ITER * 100
            prop_acc = prop_acc / PRINT_ITER

            print "[%d][%d] KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Prop: %.4f" % (epoch, it+1, kl_div, word_acc, topo_acc, assm_acc, steo_acc, prop_acc)
            
            x_plot.append(it)
            kl_plot.append(kl_div)
            word_plot.append(word_acc)
            topo_plot.append(topo_acc)
            assm_plot.append(assm_acc)
            steo_plot.append(steo_acc)
            prop_plot.append(prop_acc)
            
            word_acc,topo_acc,assm_acc,steo_acc,prop_acc = 0,0,0,0,0
            sys.stdout.flush()
            
#             if (it + 1) / PRINT_ITER == 3:
#                 break
            
    #Plot per 1 epoch
    print "Cosume Time per Epoch %s" % (str(datetime.now()-start))
    save_KL_plt(folder_name, epoch, x_plot, kl_plot)
    save_Acc_plt(folder_name, epoch, x_plot, word_plot, topo_plot, assm_plot, steo_plot)
    save_Prop_plt(folder_name, epoch, x_plot, prop_plot)
    x_plot=[]
    kl_plot=[]
    word_plot=[]
    topo_plot=[]
    assm_plot=[]
    steo_plot=[]
    prop_plot=[]

    scheduler.step()
    print "learning rate: %.6f" % scheduler.get_lr()[0]
    torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))

