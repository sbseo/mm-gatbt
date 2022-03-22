"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import os
import dgl

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.graph_datasets import MovieDataset
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h.type(dtype=torch.float32)).flatten(1)
        # output projection
        # print(h.shape)
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits

class GATEmbedder(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATEmbedder, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h.type(dtype=torch.float32)).flatten(1)
        # output projection
        # print(h.shape)
        return torch.tanh(h)

class GATEncoder(nn.Module):
    def __init__(self, args):
        super(GATEncoder, self).__init__()
        self.args = args
        self.dataset = MovieDataset(args)
        self.graph = self.dataset[0].to('cuda')
        self.graph = dgl.add_self_loop(self.graph)

        heads = torch.as_tensor(([args.num_heads] * args.num_layers) + [args.num_output_heads], dtype=torch.int32)
        self.gat = GATEmbedder(g=self.graph, 
                    num_layers=args.num_layers,
                    in_dim=args.img_infeat,
                    num_hidden=args.num_hidden, #default 10
                    num_classes=args.n_classes,
                    heads=heads,
                    activation=F.elu,
                    feat_drop=args.gat_feat_drop, #default .1
                    attn_drop=args.gat_attn_drop,
                    negative_slope=args.gat_slope,
                    residual=True
                    ).to('cuda')
        self.gat.eval()
        dic = torch.load(args.gnn_load) 
        self.gat.load_state_dict(self.preprocess(dic))

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
        
        # self.h = self.gat(self.graph, self.features)

    def forward(self, ids):
        h = self.gat(self.graph, self.features)
        out =list()
        for id in ids:
            if self.id2node.get(id.item(), -1) != -1:
                node = self.id2node[id.item()]
                out.append(h[node,:])
            else:
                out.append(torch.zeros(1, self.h.shape[1]))
        out = torch.stack(out)
        return out

    def preprocess(self, dic):
        dic2 = {}
        last_layer = self.args.num_layers
        for k in dic.keys():
            if k in [f"gat_layers.{last_layer}.attn_l", f"gat_layers.{last_layer}.attn_r", f"gat_layers.{last_layer}.bias", f"gat_layers.{last_layer}.fc.weight", f"gat_layers.{last_layer}.res_fc.weight"]:
                continue
            # if k in [f"gat_layers.{last_layer}.attn", f"gat_layers.{last_layer}.fc_src.weight", f"gat_layers.{last_layer}.fc_src.bias", f"gat_layers.{last_layer}.fc_dst.weight", f"gat_layers.{last_layer}.fc_dst.bias", f"gat_layers.{last_layer}.res_fc.bias"]:
            #     continue
            dic2[k] = dic[k]
        return dic2
