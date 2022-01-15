from dgl.nn import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

if __name__ == "__main__":
    pass
    # model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)