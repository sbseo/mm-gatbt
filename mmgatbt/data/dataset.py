#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.data_path = self.args.data_path
        if args.model in ["mmbt", "mmsagebt", "mmgatbt"]:
            self.text_start_token = ["[SEP]"]
        else:
            self.text_start_token = ["[CLS]"]  

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model in ["mmbt", "mmsagebt", "mmgatbt"]:
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data[index]["text"])[
                : (self.args.max_seq_len - 1)
            ]
        )
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt", "mmsagebt", "mmgatbt"]:
            if self.data[index]["image"]:
                im_name = self.data[index]["image"].split("/")[-1]
                im = cv2.imread(os.path.join(os.path.join(self.args.imdir_path, im_name)))
                # im = cv2.imread(os.path.join(self.imdir_path, self.data[index]["image"]))
                # width = int(im.shape[1] * .25)
                # height = int(im.shape[0] * .25)
                # im = cv2.resize(im, (width, height))
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                image =  Image.fromarray(im)

                # image = Image.open(
                #     os.path.join(self.data_path, self.data[index]["image"])
                # ).convert("RGB")
                # width, height = image.size
                # width, height = int(width * .25), int(height * .25)
                # image = image.resize((width, height))
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        if self.args.model in ["mmsagebt", "mmgatbt"]:
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1
            nid = torch.tensor(int(self.data[index]["id"]))
            return sentence, segment, image, label, nid

        if self.args.model == "gcn_bert":
            image = torch.tensor(int(self.data[index]["id"]))
            assert self.data[index]["id"] != None

        return sentence, segment, image, label
