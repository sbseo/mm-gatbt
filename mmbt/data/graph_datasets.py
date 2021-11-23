import pandas as pd
from dgl.data import DGLDataset
import numpy as np
import pandas as pd
import torch
import dgl

class MovieDataset(DGLDataset):
    def __init__(self, args):
        self.args = args
        self.vocab = args.vocab
        
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
        nodes_data = pd.read_csv('../../dataset/mmimdb/node_data.csv')
        edges_data = pd.read_csv('../../dataset/mmimdb/edge_data.csv')

        # TO DO: 
        # 1) update feature representation. let's use glove
        x = nodes_data['text'].str.split().to_list()
        x = list(map(lambda sen: ["[CLS]"] + sen, x))
        x = list(map(lambda sen: self.glove_enc(sen), x))
        node_features = torch.from_numpy(np.array(x))

        # 2) update label representation
        y = nodes_data['label'].to_list()

        node_labels = torch.zeros((len(y), self.args.n_classes))
        
        for i, row in enumerate(y):
            row = row.strip('][').split(', ')
            for j, tgt in enumerate(row):
                tgt = tgt.strip("'")
                node_labels[i, self.args.labels.index(tgt)] = 1
        
        """legacy
        node_features = torch.from_numpy(nodes_data['text'].astype('category').cat.
        des.to_numpy())
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
        n_train = int(15513)
        # n_train = int(n_nodes * 0.8)
        n_val = int(n_nodes * 0)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['test_mask'] = test_mask

        # val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # val_mask[n_train:n_train + n_val] = True
        # self.graph.ndata['val_mask'] = val_mask


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1