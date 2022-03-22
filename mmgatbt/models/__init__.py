#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from models.bert import BertClf
from models.bow import GloveBowClf
from models.concat_bert import MultimodalConcatBertClf
from models.concat_bow import  MultimodalConcatBowClf
from models.image import ImageClf
from models.mmbt import MultimodalBertClf
from models.concat_gcn_bert import MultimodalConcatGCNBertClf
from models.mmsagebt import MultimodalSageBertClf
from models.mmsagebt2 import MultimodalSageBert2Clf
from models.visualbert import VisualBertClf
from models.mmgatbt import MultimodalGATBertClf
from models.mmgatbt2 import MultimodalGAT2BertClf

MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    "mmsagebt": MultimodalSageBertClf,
    "mmsagebt2": MultimodalSageBert2Clf,
    "mmgatbt": MultimodalGATBertClf,
    "mmgatbt2": MultimodalGAT2BertClf,
    "gcn_bert": MultimodalConcatGCNBertClf,
    "visualbert": VisualBertClf,
}

def get_model(args):
    return MODELS[args.model](args)