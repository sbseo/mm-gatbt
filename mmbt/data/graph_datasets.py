import os
import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer
# from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from PIL import Image
import torchvision.transforms as transforms
import torchvision

from data.img_dataset import ImageDataset


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
    
    def add_padding(self, sen):
        while len(sen) < self.args.max_seq_len:
            sen.append(0)
        return sen

    def preprocess(self, dic):
        dic2 = {}
        for k in dic.keys():
            if "clf" in k:
                continue
            new_key = k.replace("enc.bert.", "")
            dic2[new_key] = dic[k]
        return dic2

    def process(self):
        nodes_data = pd.read_csv(os.path.join(self.args.graph_path,'node_data.csv'))
        edges_data = pd.read_csv(os.path.join(self.args.graph_path, 'edge_data.csv'))

        # 1) update feature representation. let's use glove
        if self.args.txt_enc == "glove":
            x = nodes_data['text'].str.split().to_list()
            x = list(map(lambda sen: ["[CLS]"] + sen, x))
            x = list(map(lambda sen: self.glove_enc(sen), x))
            node_features = torch.from_numpy(np.array(x))
            print(node_features.shape)
        elif self.args.txt_enc == "bert":
            tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)
            model = BertModel.from_pretrained(self.args.bert_model).cuda()
            best_params = torch.load("../bert_base/model_best.pt")
            best_params = self.preprocess(best_params["state_dict"])
            model.load_state_dict(best_params)
            model.eval()
            # tokenized = nodes_data['text'].apply((lambda sen: tokenizer.encode_plus(sen, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = False, return_tensors = "pt")))
            # print(tokenized.values)
            
            # x = np.array(tokenized.values)
            x = list()
            attention_mask = list()
            for sen in nodes_data['text'].tolist():
                dic = tokenizer.encode_plus(sen, max_length=512, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
                x.append(dic['input_ids'])
                attention_mask.append(dic['attention_mask'])

            x = torch.stack(x).squeeze()
            print(x, x.shape)
            attention_mask = torch.stack(attention_mask).squeeze()
            # attention_mask = attention_mask.cuda()
            # x = x.cuda()
            # TO DO: use dataloader 
            data = TensorDataset(x, attention_mask)
            data_sampler = SequentialSampler(data)
            data_loader = DataLoader(data, sampler=data_sampler, batch_size=self.args.batch_sz)
            
            node_features = list()
            with torch.no_grad():
                for batch in data_loader:
                    x, attn_mask = batch
                    x = x.cuda()
                    attn_mask = attn_mask.cuda()
                    output, _ = model(x, attention_mask=attn_mask, output_all_encoded_layers=False)
                    # feat = output['last_hidden_state'][:,0,:]
                    feat = output[:,0,:]
                    node_features.append(feat.detach().cpu())
            node_features = torch.cat(node_features)
            print(node_features.shape)


        # 1-2) img
        # Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.46777044, 0.44531429, 0.40661017],
                    std=[0.12221994, 0.12145835, 0.14380469],
                ),
            ]
        )
        if self.args.img_enc != 'none':
            im_dataset = ImageDataset(self.args, nodes_data)
            data_loader = DataLoader(im_dataset, batch_size=16, shuffle=False, num_workers=12)
            
            # this is typo. res should be changed to mobile.
            if self.args.img_enc == 'res':
                model = torchvision.models. mobilenet_v3_small(pretrained=True)
            elif self.args.img_enc == 'eff':
                model = torchvision.models.efficientnet_b4(pretrained=True)
            elif self.args.img_enc == 'eff6':
                model = torchvision.models.efficientnet_b6(pretrained=True)
            elif self.args.img_enc == 'resnet152':
                model = torchvision.models.resnet152(pretrained=True)
                
            model.eval()
            # model = model.to('cuda')
            node_features = list()
            with torch.no_grad():
                for batch in data_loader:
                    im = batch
                    # feat.shape: torch.Size([16, 1000])
                    feat = model(im)
                    node_features.append(feat.unsqueeze(0))
            # im_data = TensorDataset(imgs)
            # data_sampler = SequentialSampler(im_data)
            # im_data_loader = DataLoader(im_data, sampler=data_sampler,num_workers=16, batch_size=self.args.batch_sz)
            
            
            
            # imgs = [torch.tensor([i]) for i in range(len(nodes_data['image']))]
            # imgs = torch.stack(imgs)
            # print(imgs.shape)
            # im_data = TensorDataset(imgs)
            # data_sampler = SequentialSampler(im_data)
            # im_data_loader = DataLoader(im_data, sampler=data_sampler,num_workers=16, batch_size=self.args.batch_sz)
            # model = torchvision.models.resnet152(pretrained=True)
            # node_features = list()
            # model = model.to('cuda')
            # with torch.no_grad():
            #     for batch in im_data_loader:
            #         for x in batch[0]:
            #             x = x.item()
            #             img_path = nodes_data['image'][x]
            #             img = Image.open(
            #                 os.path.join(self.args.data_path, img_path)
            #             ).convert("RGB")
            #             img = transform(img)
            #             img = img.unsqueeze(0)
            #             img = img.to('cuda')
            #             feat = model(img)
            #             node_features.append(feat.unsqueeze(0))

            # model = torchvision.models.resnet152(pretrained=True).cuda()
            # for i, x in enumerate(imgs):
            #     print(i)
            #     output = model(x.cuda())
            #     node_features.append(output.detach().cpu())
            node_features = torch.cat(node_features, dim=1)
            node_features = node_features.squeeze(0)
            print(node_features.shape)

        print("complete")
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
        if self.args.weight:
            edge_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())

        # print(edges_src)
        # print(edges_dst)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        if self.args.weight:
            self.graph.edata['weight'] = edge_features

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



class MovieDataset_dev(DGLDataset):
    def __init__(self, args):
        self.args = args
        self.vocab = args.vocab
        
        super().__init__(name='movie_dataset_dev')


    def glove_enc(self, sen):
        sen = sen[:self.args.max_seq_len - 1]
        sen = [self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"] for w in sen]
        while len(sen) < self.args.max_seq_len:
            sen.append(0)
        return sen
    
    def add_padding(self, sen):
        while len(sen) < self.args.img_infeat:
            sen.append(0)
        return sen

    def preprocess(self, dic):
        dic2 = {}
        for k in dic.keys():
            if "clf" in k:
                continue
            new_key = k.replace("enc.bert.", "")
            dic2[new_key] = dic[k]
        return dic2

    def process(self):
        nodes_data = pd.read_csv(os.path.join(self.args.graph_path,'node_data.csv'))
        edges_data = pd.read_csv(os.path.join(self.args.graph_path, 'edge_data.csv'))

        # 1) update feature representation. let's use bert
        x = nodes_data['text'].str.split().to_list()
        x = list(map(lambda sen: ["[CLS]"] + sen, x))
        x = list(map(lambda sen: self.glove_enc(sen), x))
        x = list(map(lambda sen: self.add_padding(sen), x))
        node_features = torch.from_numpy(np.array(x))
        print(node_features.shape)
        
        print("complete")
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
        if self.args.weight:
            edge_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())

        # print(edges_src)
        # print(edges_dst)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        if self.args.weight:
            self.graph.edata['weight'] = edge_features

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
