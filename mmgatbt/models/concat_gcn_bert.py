#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from models.bert import BertEncoder
from models.image import ImageEncoder
from models.graphsage import SageEncoder


class MultimodalConcatGCNBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatGCNBertClf, self).__init__()
        self.args = args
        self.txtenc = BertEncoder(args)
        self.genc = SageEncoder(args)

        last_size = args.hidden_sz + (args.img_hidden_sz * args.num_image_embeds)
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            if args.include_bn:
                self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden

        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, txt, mask, segment, img):
        txt = self.txtenc(txt, mask, segment)
        # print(img)
        img = self.genc(img).cuda()
        
        # out = txt
        # print(txt.shape)
        out = torch.cat([txt, img], -1)
        for layer in self.clf:
            out = layer(out)
        return out


# python3 mmbt/train.py --batch_sz 8 --gradient_accumulation_steps 40  --savedir ./ --name gcn_bert_dev --bert_model bert-base-uncased --data_path ../dataset/  --task mmimdb --task_type multilabel  --model gcn_bert --freeze_txt 5 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --img_hidden_sz 0