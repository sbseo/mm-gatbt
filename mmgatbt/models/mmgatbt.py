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
from pytorch_pretrained_bert.modeling import BertModel
from scipy.ndimage import gaussian_filter, laplace

from models.gat import GATEncoder
from models.image import ImageEncoder


# takes txt_embedding as argument
class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)

        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        if self.args.pos:
            embeddings = token_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = token_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalGATBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalGATBertEncoder, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        args.img_hidden_sz = args.g_hidden_sz
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler

        self.genc = GATEncoder(args)

    def forward(self, input_txt, attention_mask, segment, input_img, nid):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )

        # [8,200] -> [8, 3, 576]
        gembed = self.genc(nid).cuda()
        if self.args.num_image_embeds==1:
            glist = [gembed] * self.args.num_image_embeds
        # experimental function !!!
        elif self.args.num_image_embeds==3:
            gembed_laplace = laplace(gembed.cpu().detach().numpy())
            gembed_gaussian = gaussian_filter(gembed.cpu().detach().numpy(), sigma=1)
            glist = [gembed, torch.tensor(gembed_gaussian).cuda(), torch.tensor(gembed_laplace).cuda()]

        gembed = torch.stack(glist)
        gembed = gembed.permute(1,0,2)

        node_embed_out = self.img_embeddings(gembed, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([node_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID
        
        encoded_layers = self.encoder(
            encoder_input, extended_attention_mask, output_all_encoded_layers=False
        )

        return self.pooler(encoded_layers[-1])

class MultimodalGATBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalGATBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalGATBertEncoder(args)
        last_size = args.hidden_sz
        self.clf = nn.Linear(last_size, args.n_classes)

    def forward(self, txt, mask, segment, img, nid):
        mmgatbt = self.enc(txt, mask, segment, img, nid)

        return self.clf(mmgatbt)
