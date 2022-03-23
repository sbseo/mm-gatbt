import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch._C import dtype
from torch.functional import Tensor

def train(g, model, args, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    best_val_acc = 0
    best_test_acc = 0
    best_test_acc_macro = 0
    
    features = g.ndata['feat'].to('cuda')
    labels = g.ndata['label'].to('cuda')
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    for e in range(args.epoch):
        # Forward
        model.train()
        logits = model(g, features)

        # Compute prediction
        pred = torch.sigmoid(logits)

        # Note that you should only compute the losses of the nodes in the training set.
        freqs = [args.label_freqs[l] for l in args.labels]
        label_weights = (torch.FloatTensor(freqs) / 23292) ** -1 # 23
        criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda()).to('cuda')
        loss = criterion(pred[train_mask], labels[train_mask])
 
        with torch.no_grad():   
            model.eval()
            y = labels.cpu().detach().numpy() == 1
            y_hat = pred.cpu().detach().numpy() > args.threshold
            train_acc = f1_score(y[train_mask.cpu()], y_hat[train_mask.cpu()], average="micro")
            test_acc = f1_score(y[test_mask.cpu()], y_hat[test_mask.cpu()], average="micro")
            test_acc_macro = f1_score(y[test_mask.cpu()], y_hat[test_mask.cpu()], average="macro", zero_division=0)

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_test_acc_macro = test_acc_macro
                if args.save_model:
                    save_name = str(args.name)+".pth"
                    torch.save(model.state_dict(), save_name)
                    
                    # save labels
                    f = open("./{}_pred.txt".format(args.name), "w")
                    f2 = open("./{}_true.txt".format(args.name), "w")
                    
                    for sen in y_hat[test_mask.cpu()]:
                        sen = " ".join([str(num) for num in list(map(lambda x: int(x), sen))])
                        f.write(f"{sen} \n")
                    for sen in y[test_mask.cpu()]:
                        sen = " ".join([str(num) for num in list(map(lambda x: int(x), sen))])
                        f2.write(f"{sen} \n")
                    f.close()
                    f2.close()

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