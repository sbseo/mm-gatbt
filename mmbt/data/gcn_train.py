import torch
from torch._C import dtype
from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def CXE(predicted, target):
    return -(target * np.log(predicted)).sum(dim=1).mean()

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(1000):
        # Forward
        logits = model(g, features)

        # Compute prediction
        # pred = logits.argmax(1)
        pred = torch.sigmoid(logits)

        # print(pred)
        # print(labels)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # loss = nn.KLDivLoss(pred[train_mask], labels[train_mask])
        # print(pred[train_mask].shape)
        # print(labels[train_mask].shape)
        # loss = CXE(pred[train_mask], labels[train_mask])
        # loss = nn.BCEWithLogitsLoss(pred[train_mask], labels[train_mask])
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(pred[train_mask], labels[train_mask])
            # Compute accuracy on training/validation/test
            # train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        with torch.no_grad():   
            model.eval()
            y = pred.detach().numpy()
            y_hat = labels.detach().numpy()
            y = np.vstack(labels)
            y_hat = np.vstack(pred)
            y = y==1
            y_hat = y_hat == 1

            train_acc = f1_score(y[train_mask], y_hat[train_mask], average="macro")
            val_acc = f1_score(y[val_mask], y_hat[val_mask], average="macro")
            test_acc = f1_score(y[test_mask], y_hat[test_mask], average="macro")

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))