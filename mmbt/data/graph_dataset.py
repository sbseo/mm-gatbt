import pandas as pd
import dgl
from dgl.data import DGLDataset
import torch
import os

import numpy as np
import torch.nn as nn

import argparse
from collections import Counter
import json

class Vocab(object):
    def __init__(self, emptyInit=False):
        if emptyInit:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)

    def add(self, words):
        cnt = len(self.itos)
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w] = cnt
            self.itos.append(w)
            cnt += 1
        self.vocab_sz = len(self.itos)

def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list



def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="../dataset/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="../gloVe/glove.6B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="bow", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)

class GloveBowEncoder(nn.Module):
    def __init__(self, args):
        super(GloveBowEncoder, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_sz, args.embed_sz)
        self.load_glove()
        self.embed.weight.requires_grad = False

    def load_glove(self):
        print("Loading glove")
        pretrained_embeds = np.zeros(
            (self.args.vocab_sz, self.args.embed_sz), dtype=np.float32
        )
        for line in open(self.args.glove_path):
            w, v = line.split(" ", 1)
            if w in self.args.vocab.stoi:
                pretrained_embeds[self.args.vocab.stoi[w]] = np.array(
                    [float(x) for x in v.split()], dtype=np.float32
                )
        self.embed.weight = torch.nn.Parameter(torch.from_numpy(pretrained_embeds))

    def forward(self, x):
        return self.embed(x).sum(1)



class MovieDataset(DGLDataset):
    def __init__(self, args):
        self.args = args
        self.vocab = args.vocab
        # self.txtenc = GloveBowEncoder(args)
        # self.n_classes = 

        super().__init__(name='movie_dataset')

    def glove_enc(self, sen):
        sen = sen[:self.args.max_seq_len - 1]
        sen = [self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"] for w in sen]
        while len(sen) < self.args.max_seq_len:
            sen.append(0)
        return sen

    def process(self):
        # nodes_data = pd.read_csv('../members.csv')
        # edges_data = pd.read_csv('../interactions.csv')
        nodes_data = pd.read_csv('../dataset/mmimdb/node_data.csv')
        edges_data = pd.read_csv('../dataset/mmimdb/edge_data.csv')

        # TO DO: 
        # 1) update feature representation. let's use glove
        x = nodes_data['text'].str.split().to_list()
        x = list(map(lambda sen: ["[CLS]"] + sen, x))
        x = list(map(lambda sen: self.glove_enc(sen), x))
        node_features = torch.from_numpy(np.array(x))

        # 2) update label representation
        y = nodes_data['label'].to_list()

        node_labels = torch.zeros((len(y), args.n_classes))
        
        for i, row in enumerate(y):
            row = row.strip('][').split(', ')
            for j, tgt in enumerate(row):
                tgt = tgt.strip("'")
                node_labels[i, self.args.labels.index(tgt)] = 1
        
        """legacy
        node_features = torch.from_numpy(nodes_data['text'].astype('category').cat.codes.to_numpy())
        node_labels = torch.from_numpy(nodes_data['label'].astype('category').cat.codes.to_numpy())
        """
        print(node_features)
        print(node_labels)

        #  edge is complete
        # edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())

        # print(edges_src)
        # print(edges_dst)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        # self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    
    vocab = Vocab()
    word_list = get_glove_words(args.glove_path)
    vocab.add(word_list)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "dev.jsonl")
    )
    args.n_classes = len(args.labels)

    dataset = MovieDataset(args)
    graph = dataset[0]
    print(graph)




