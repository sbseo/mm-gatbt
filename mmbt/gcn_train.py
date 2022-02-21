import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch._C import dtype
from torch.functional import Tensor


def CXE(predicted, target):
    return -(target * np.log(predicted)).sum(dim=1).mean()

def train(g, model, args, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # 0.00001 works good
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    best_val_acc = 0
    best_test_acc = 0
    best_test_acc_macro = 0
    
    # save_path = os.path.join(args.savedir, args.name)
    # os.makedirs(save_path)

    features = g.ndata['feat'].to('cuda')
    labels = g.ndata['label'].to('cuda')
    train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(args.epoch):
        # Forward
        model.train()
        logits = model(g, features)

        # Compute prediction
        # pred = logits.argmax(1)
        
        """ Needs double check here && two classes"""
        # print(logits[0])
        pred = torch.sigmoid(logits)
        # print(pred[5])


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
        freqs = [args.label_freqs[l] for l in args.labels]
        label_weights = (torch.FloatTensor(freqs) / 23292) ** -1 # 23
        criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda()).to('cuda')
        # criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(pred[train_mask], labels[train_mask])
            # Compute accuracy on training/validation/test
            # train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        with torch.no_grad():   
            model.eval()
            y = pred.cpu().detach().numpy() == 1
            y_hat = labels.cpu().detach().numpy() > args.threshold
            # y = np.vstack(labels)
            # y_hat = np.vstack(pred)
            # y = y==1
            # print(y_hat[0])
            # y_hat = y_hat > .25 # 0.25
            # print(y[5])
            # print(y_hat[5])
            train_acc = f1_score(y[train_mask.cpu()], y_hat[train_mask.cpu()], average="micro")
            # val_acc = f1_score(y[val_mask], y_hat[val_mask], average="micro")
            test_acc = f1_score(y[test_mask.cpu()], y_hat[test_mask.cpu()], average="micro")
            test_acc_macro = f1_score(y[test_mask.cpu()], y_hat[test_mask.cpu()], average="macro", zero_division=0)

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_test_acc < test_acc:
                # best_val_acc = val_acc
                best_test_acc = test_acc
                best_test_acc_macro = test_acc_macro
                # torch.save(model.state_dict(), save_path)
                if args.save_model:
                    save_name = str(args.name)+".pth"
                    torch.save(model.state_dict(), save_name)

                # save labels
                # f = open("./{}_lr-{}_pred.txt".format(args.model, args.lr), "w")
                # f2 = open("./{}_lr-{}_true.txt".format(args.model, args.lr), "w")
                
                # for sen in y_hat[test_mask.cpu()]:
                #     f.write(str(sen))
                #     f.write("\n")
                # for sen in y[test_mask.cpu()]:
                #     f2.write(str(sen))
                #     f2.write("\n")
                
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.scheduler:
            scheduler.step(test_acc)
       
        if e % 100 == 0:
            print('In epoch {}, loss: {:.3f}, train acc: {:.3f}, test micro: {:.3f}, test macro: {:.3f}, (best {:.3f}, {:.3f})'.format(
                e, loss, train_acc, test_acc, test_acc_macro, best_test_acc, best_test_acc_macro))
            if args.save_model:
                logger.write('In epoch {}, loss: {:.3f}, train acc: {:.3f}, test micro: {:.3f}, test macro: {:.3f}, (best {:.3f}, {:.3f})'.format(
                e, loss, train_acc, test_acc, test_acc_macro, best_test_acc, best_test_acc_macro) + "\n")