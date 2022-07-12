# MM-GATBT: Enriching Multimodal Representation Using Graph Attention Network 

<img src="./fig_model.png" alt="model" width=750 />


- This repository contains implementation of MM-GATBT published at NAACL 2022 SRW. 
- MM-GATBT is built upon [MMBT](https://github.com/facebookresearch/mmbt)


## Installation 

> Clone this repository into your local directory 

```
git clone git@github.com:sbseo/mm-gatbt.git
```

```
cd mm-gatbt
```


##  Requirements

1. Software requirement

``` 
pip install -r requirements.txt
```

2. Hardware requirement (Optional)

   - GPU: Nvidia 3090 (vRAM: 24 gb)
   - Driver Version: 470.103.01

 

## MM-IMDb Dataset

1. Download dataset (8.1G) (Arevalo et al., 2017)
```
wget -O mmimdb.tar.gz https://archive.org/download/mmimdb/mmimdb.tar.gz
```

2. Decompress the file after download.
```
tar -xf mmimdb.tar.gz
```

3. Preprocess dataset (Kiela, 2019)
```
cd scripts
```
```
python3 format_mmimdb_dataset.py ../
```
4. Construct graph (This may take awhile)
```
python3 format_mmimdb_as_graph.py ../ medium
```
```
cd ..
```

## Train model

Pre-saved EfficientNet embedding is available to reduce image loading time. If you prefer to load EfficientNet from scratch, simply remove `load_imgembed`  argument.

   - Pre-saved EfficientNet embedding: [eff_embedding.pt](https://drive.google.com/file/d/1wHsqBQfeXqGf_xEQRO7GIr7aJlkFY3bk/view?usp=sharing)

```
gdown 1wHsqBQfeXqGf_xEQRO7GIr7aJlkFY3bk
```

1. Train image-based GAT

> Training GAT will save its best model `eff_gat_256.pth` and its prediction results under dir `./eff_gat_256/`

    python3 mmgatbt/gnn_train.py --img_enc eff --model gat --name eff_gat_256 --load_imgembed ./eff_embedding.pt



2. Train MM-GATBT

> Training MM-GATBT will save its prediction results under dir `./mmgatbt_eff256/`

    python3 mmgatbt/train.py --img_enc eff --model mmgatbt --name mmgatbt_eff256 --gnn_load ./eff_gat_256/eff_gat_256.pth --batch_sz 12 --load_imgembed ./eff_embedding.pt


## Pre-trained Model

Pre-trained models for both MM-GATBT (main model) and image-based GAT (submodel) are available. 

- Image-based GAT: [eff_gat_256.pth](https://drive.google.com/file/d/1S4ltCiWou75qKYmXnRmxU-2py0Oz6Czb/view?usp=sharing)
- MM_GATBT: [mmgatbt_eff256.zip](https://drive.google.com/file/d/12O9-kOBxk-Ggw85Vo9M5SDFcTpDPlMcE/view?usp=sharing)

```
gdown 1S4ltCiWou75qKYmXnRmxU-2py0Oz6Czb
```
```
gdown 12O9-kOBxk-Ggw85Vo9M5SDFcTpDPlMcE
unzip mmgatbt_eff256.zip
```

## Validation 

Set `max_epochs` to `0` for validation

> Predicted results can be also found under `./mmgatbt_eff256/`

    python3 mmgatbt/train.py --img_enc eff --model mmgatbt --name mmgatbt_eff256 --gnn_load ./eff_gat_256.pth --batch_sz 12 --max_epochs 0 --load_imgembed ./eff_embedding.pt


## Citation
```
@inproceedings{seo-etal-2022-mm,
    title = "{MM}-{GATBT}: Enriching Multimodal Representation Using Graph Attention Network",
    author = "Seo, Seung Byum  and
      Nam, Hyoungwook  and
      Delgosha, Payam",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Student Research Workshop",
    month = jul,
    year = "2022",
    address = "Hybrid: Seattle, Washington + Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-srw.14",
    pages = "106--112",
    abstract = "While there have been advances in Natural Language Processing (NLP), their success is mainly gained by applying a self-attention mechanism into single or multi-modalities. While this approach has brought significant improvements in multiple downstream tasks, it fails to capture the interaction between different entities. Therefore, we propose MM-GATBT, a multimodal graph representation learning model that captures not only the relational semantics within one modality but also the interactions between different modalities. Specifically, the proposed method constructs image-based node embedding which contains relational semantics of entities. Our empirical results show that MM-GATBT achieves state-of-the-art results among all published papers on the MM-IMDb dataset.",
}
```
