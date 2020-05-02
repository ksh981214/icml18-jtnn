#-*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree
import numpy as np
from jtnn_enc import JTNNEncoder
from mpn import MPN
from jtmpn import JTMPN
import cPickle as pickle
import os, random

import numpy as np

import sys
import gc

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class MolTreeFolder(object):
    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            print("data ref cnt: {}".format(sys.getrefcount(data)))
            #print(gc.get_referents(data))

            print("batches ref cnt: {}".format(sys.getrefcount(batches)))
            #print(gc.get_referents(batches))

            print("dataset ref cnt: {}".format(sys.getrefcount(dataset)))
            #print(gc.get_referents(dataset))

            print("dataloader ref cnt: {}".format(sys.getrefcount(dataloader)))
            #print(gc.get_referents(dataloader))

            #del data, batches, dataset, dataloader

            del data, dataloader
            #gc.collect()

            # print("dataset ref cnt: {}".format(sys.getrefcount(dataset)))
            del dataset
            #gc.collect()

            # print("batches ref cnt: {}".format(sys.getrefcount(batches)))
            del batches

            # try:
            #     print("data ref cnt: {}".format(sys.getrefcount(data)))
            # except Exception as e:
            #     print(e)
            #     pass
            #
            # try:
            #     print("dataloader ref cnt: {}".format(sys.getrefcount(dataloader)))
            # except Exception as e:
            #     print(e)
            #     pass
            #
            # try:
            #     print("dataset ref cnt: {}".format(sys.getrefcount(dataset)))
            # except Exception as e:
            #     print(e)
            #     pass
            #
            # try:
            #     print("batches ref cnt: {}".format(sys.getrefcount(batches)))
            # except Exception as e:
            #     print(e)
            #     pass

            # print("af data ref cnt: {}".format(sys.getrefcount(data)))
            # print(gc.get_referents(data))

            # print("af batches ref cnt: {}".format(sys.getrefcount(batches)))
            # print(gc.get_referents(batches)[:10])
            #
            # print("af dataset ref cnt: {}".format(sys.getrefcount(dataset)))
            # print(gc.get_referents(dataset):[:10])

            # print("af dataloader ref cnt: {}".format(sys.getrefcount(dataloader)))
            # print(gc.get_referents(dataloader))

            # gc.collect()

            # print("gc af data ref cnt: {}".format(sys.getrefcount(data)))
            # print(gc.get_referents(data))

            # print("gc af batches ref cnt: {}".format(sys.getrefcount(batches)))
            # print(gc.get_referents(batches)[:10])
            #
            # print("gc af dataset ref cnt: {}".format(sys.getrefcount(dataset)))
            # print(gc.get_referents(dataset)[:10])

            # print("gc af dataloader ref cnt: {}".format(sys.getrefcount(dataloader)))
            # print(gc.get_referents(dataloader))

class MolTreeFolderMLP(object):
    def __init__(self, data_folder, gene_exp, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        path = os.path.join(gene_exp,'train_embedding_121.txt')
        print(path)
        with open(path) as f:
            all_gene=f.readlines()

        print("Finish gene exp loading")

        self.all_gene = [list(map(float, emb.split())) for emb in all_gene] #LIST[LIST]
        print("Len(all_gen) is {}".format(len(self.all_gene)))


        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        idx =0
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            gene = self.all_gene[idx:idx+len(data)]
            idx = idx + len(data)

            if len(data) != len(gene):
                print("len(data) != len(gene)")

            if self.shuffle:
                #random.shuffle(data) #shuffle data before batch
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                data = np.array(data)[indices]
                data = list(data)

                gene = np.array(gene)[indices]
                gene = list(gene)

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            gene_batches = [gene[i : i + self.batch_size] for i in xrange(0, len(gene), self.batch_size)]
            if len(gene_batches[-1]) < self.batch_size:
                gene_batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for i,b in enumerate(dataloader):
                #print("len(gene_batches[i]) is {}".format(len(gene_batches[i])))
                yield b, gene_batches[i]

            del data, batches, dataset, dataloader, gene, gene_batches

class MolTreeFolderMJ(object):
    def __init__(self, data_folder, neg_data_folder, gene_exp, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.neg_data_folder = neg_data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.neg_data_files = [fn for fn in os.listdir(neg_data_folder)]
        
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        
        #GeneExpression Load
        with open(gene_exp) as f:
            all_gene=f.readlines()
        print("Finish gene exp loading")
        self.all_gene = [list(map(float, emb.split())) for emb in all_gene] #LIST[LIST]
        print("Len(all_gen) is {}".format(len(self.all_gene)))
    
        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        idx =0
        for i, fn in enumerate(self.data_files):
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)
            
            #negative mols
            neg_fn = os.path.join(self.neg_data_folder, self.neg_data_files[i])
            with open(neg_fn) as neg_f:
                neg_data = pickle.load(neg_f)

            gene = self.all_gene[idx:idx+len(data)]
            idx = idx + len(data)

            if len(data) != len(gene):
                print("len(data) != len(gene)")
                
            if len(neg_data) != len(gene):
                print("len(neg_data) != len(gene)")

            if self.shuffle:
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                
                #pos:neg = 1:5
                pos_idx = list(indices)[:int(len(data)/6)]
                neg_idx = list(indices)[int(len(data)/6):]
                
                #make label
                pos_label = np.ones(len(pos_idx), dtype=np.float)
                neg_label = np.zeros(len(neg_idx), dtype=np.float)
                label = np.append(pos_label, neg_label)

                pos_data = np.array(data)[pos_idx]
                neg_data = np.array(neg_data)[neg_idx]
                data = np.append(pos_data, neg_data)

                gene = np.array(gene)[indices]

                #one more shuffle
                indices = np.arange(len(data))
                np.random.shuffle(indices)
    
                data = list(data[indices])
                gene = list(gene[indices])
                label= list(label[indices])
                
                del indices

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            gene_batches = [gene[i : i + self.batch_size] for i in xrange(0, len(gene), self.batch_size)]
            if len(gene_batches[-1]) < self.batch_size:
                gene_batches.pop()
                
            label_batches = [label[i : i + self.batch_size] for i in xrange(0, len(label), self.batch_size)]
            if len(label_batches[-1]) < self.batch_size:
                label_batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for i,b in enumerate(dataloader):
                yield b, gene_batches[i], label_batches[i]

            del data, batches, dataset, dataloader, gene, gene_batches, label, label_batches

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
