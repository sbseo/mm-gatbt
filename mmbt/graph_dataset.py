import argparse
import os

import dgl
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from data.graph_datasets import MovieDataset
from data.helpers import get_glove_words, get_labels_and_frequencies
from data.vocab import Vocab
from gcn_train import train
from models.gat import GAT
from models.gcn import GCN
from models.graphsage import GraphSAGE


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--txt_enc", type=str, default="glove", choices=["glove", "bert"])
    parser.add_argument("--img_enc", type=str, default="none", choices=["res", "eff", "eff6", "none"])
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["prajjwal1/bert-tiny","bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="../../dataset/")
    parser.add_argument("--graph_path", type=str, default="../../dataset/mmimdb/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="../../gloVe/glove.6B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=200)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5000)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "sage"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="./checkpoint/",
                        help="/path/to/save_dir/")                    
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=80000)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--scheduler", type=bool, default=False)
    parser.add_argument("--aggregator", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")                    
    parser.add_argument("--gat_attn_drop", type=float, default="0.1")                    
    parser.add_argument("--gat_feat_drop", type=float, default="0.1")                    
    parser.add_argument("--gat_slope", type=float, default="0.1")                    
    parser.add_argument("--save_model", type=bool, default=False)                    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    
    vocab = Vocab()
    if args.txt_enc == "glove":
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)
    elif args.txt_enc == "bert":
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz


    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl")
    )
    args.n_classes = len(args.labels)
    dataset = MovieDataset(args)
    graph = dataset[0].to('cuda')
    print(args)
    print(graph)

    # Create the model with given dimensions
    if args.model == "gcn":
        model = GCN(graph.ndata['feat'].shape[1], args.hidden_sz, args.n_classes).to('cuda')
        graph = dgl.add_self_loop(graph)
    if args.model == "sage":
        # dropout default 5e-4
        graph = dgl.add_self_loop(graph)
        model = GraphSAGE(graph.ndata['feat'].shape[1], args.hidden_sz, args.n_classes, 1, F.relu, args.dropout, args.aggregator).to('cuda')

    elif args.model == "gat":
        num_heads = 8
        num_layers = 1
        num_output_heads = 1
        heads = torch.as_tensor(([num_heads] * num_layers) + [num_output_heads], dtype=torch.int32)
        graph = dgl.add_self_loop(graph)
        model = GAT(g=graph, 
                    num_layers=num_layers,
                    in_dim=graph.ndata['feat'].shape[1],
                    num_hidden=10,
                    num_classes=args.n_classes,
                    heads=heads,
                    activation=F.elu,
                    feat_drop=args.gat_feat_drop, #default .1
                    attn_drop=args.gat_attn_drop,
                    negative_slope=args.gat_slope,
                    residual=True
                    ).to('cuda')
    print(model)

    train(graph, model, args)