import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
from data.graph_datasets import MovieDataset



class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs.type(dtype=torch.float32))
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class SageEmbedder(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(SageEmbedder, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))

    def forward(self, graph, inputs):
        h = self.dropout(inputs.type(dtype=torch.float32))
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            # if l != len(self.layers) - 1:
                # h = self.activation(h)
                # h = self.dropout(h)
        return torch.tanh(h)

class SageEncoder(nn.Module):
    def __init__(self, args):
        super(SageEncoder, self).__init__()
        self.args = args
        self.sage = SageEmbedder(args.img_infeat,
                 args.img_hidden_sz,
                 25,
                 1,
                 F.relu,
                 0,
                 'gcn').cuda()
        self.sage.eval()
        dic = torch.load(args.gnn_load) #./mmbt/sage_vico.pth
        self.sage.load_state_dict(self.preprocess(dic))
        self.dataset = MovieDataset(args)
        self.graph = self.dataset[0].to('cuda')
        self.graph = dgl.add_self_loop(self.graph)

        self.features = self.graph.ndata['feat'].cuda()
        self.labels = self.graph.ndata['label'].cuda()
        self.train_mask = self.graph.ndata['train_mask'].cuda()
        self.test_mask = self.graph.ndata['test_mask'].cuda()

        self.id2node = dict()
        f = open(os.path.join(args.graph_path, "idx2node.csv"))
        for i, sen in enumerate(f):
            if i==0:
                continue
            id, node = sen.split(",")
            self.id2node[int(id)] = int(node)

    def forward(self, ids):
        h = self.sage(self.graph, self.features)
        out =list()
        for id in ids:
            if self.id2node.get(id.item(), -1) != -1:
                node = self.id2node[id.item()]
                out.append(h[node,:])
            else:
                out.append(torch.zeros(1, self.args.img_hidden_sz))
        out = torch.stack(out)
        return out

    def preprocess(self, dic):
        dic2 = {}
        for k in dic.keys():
            if k in ["layers.1.bias", "layers.1.fc_neigh.weight"]:
                continue
            dic2[k] = dic[k]
        return dic2
