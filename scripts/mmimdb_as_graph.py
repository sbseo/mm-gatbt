###
# 1) mmimdb graph dataset should not be splited into train/test dataset
# 2) all node numbers should start from 0. That is, remap all id numbers.
### 


from collections import defaultdict
import sys
import os
import json
import pandas as pd

def format_mmimdb_add_staff(dataset_root_path, file_name, save_name):
    """return dictionary which includes movie info and staff.
    
    Requirement:
        `train.jsonl` created by mmimdb.py

    Args:
        dataset_root_path (str): root path mm-imdb dataset

    Returns:
        dobj: dictinary object where key is movie id and value is dictionary
    """

    if save_name == "sparse":
        features = ['director', 'producer', 'writer']
    elif save_name in ["medium", "medium_weight"]:
        features = ['director', 'producer', 'writer', 'cinematographer', 'art director']
    elif save_name == 'dense':
        features = ['director', 'producer', 'writer', 'cinematographer', 'art director', 'assistant director', 'editor']
    elif save_name == 'writer':
        features = ['writer']
    elif save_name == 'producer':
        features = ['producer']
    elif save_name == 'director':
        features = ['director']
    
    dobj = defaultdict()
    f = open(os.path.join(dataset_root_path, file_name), 'r')
    files = os.listdir(os.path.join(dataset_root_path, "dataset"))

    for sen in f:
        data = json.loads(sen)
        dobj[data['id']] = data

    for file_name in files:
        idx = file_name.split(".")[0]
        if "json" in file_name and dobj.get(idx, None):
            file_path = os.path.join(dataset_root_path, "dataset", file_name)
            _f = open(file_path)
            _data = json.load(_f)
            
            name_list = list()
            title_names = {feat: _data.get(feat, []) for feat in features}
            for title, l in title_names.items():
                for d in l:
                    name_list.append(d['name'])

            names = {'staff':name_list}
            dobj[idx].update(names)

    return dobj

def format_mmimdb_add_edge(dic):
    """ Find staff in all other dataset. If movies share the same staff, then connect.

    Args:
        dict (dictionary): dictionary object where key is index and value is dictionary

    Returns:
        dic: dictionary object where edges are added
    """

    for idx, d in dic.items():
        staff = set(d["staff"])
        connected = list()
        weight = 0
        for idx2, d2 in dic.items():
            if idx == idx2:
                continue
            if set.intersection(staff, set(d2["staff"])):
                connected.append(idx2)
                weight = len(set.intersection(staff, set(d2["staff"])))
        edges = {"edges":[connected, weight]}
        dic[idx].update(edges)

    return dic

def save_mmimdb_dataset_to_csv(dic, dataset_root_path, save_name):
    """Save to csv

    Args:
        dic (dict): {id:{id, feat, edges, lbl}}

    Returns:
        node_data.csv: id | feat | lbl
        edge_data.csv: src | dest
        idx2node.csv: id | node#
    """

    dic_flatten = defaultdict()
    edges = list()
    idx2node = {v:i for i, v in enumerate(dic.keys())}
    features = ["id", "text", "image", "label"]

    for idx, d in dic.items():
        node_src = idx2node[idx]
        dic_flatten[node_src] = [d[feat] for feat in features]
        # d["edges"]: [['0303970', '0098691', '2495118', '0845464', '0251433'], 1]
        # edges += [{"src": node_src, "dst": idx2node[dst[0]], 'weight': d["edges"][1]} for dst in d["edges"]] 
        edges += [{"src": node_src, "dst": idx2node[dst], 'weight': d["edges"][1]} for dst in d["edges"][0]] 

    node_data = pd.DataFrame.from_dict(dic_flatten, orient='index')
    node_data.to_csv(os.path.join(dataset_root_path, save_name + "_node_data.csv"), index=False, header=features)

    edge_data = pd.DataFrame.from_records(edges)
    edge_data.to_csv(os.path.join(dataset_root_path, save_name + "_edge_data.csv"), index=False)

    idx2node_data = pd.DataFrame.from_dict(idx2node, orient='index')
    idx2node_data.to_csv(os.path.join(dataset_root_path, save_name + "_idx2node.csv"), index=True, index_label="idx", header=["node"])

    return dic_flatten, edges, idx2node


if __name__=="__main__":
    # python3 scripts/mmimdb_as_graph.py ../dataset/mmimdb/ unsplitted.jsonl
    # python3 scripts/mmimdb_as_graph.py ../dataset/mmimdb/ unsplitted.jsonl sparse
    # python3 scripts/mmimdb_as_graph.py ../dataset/mmimdb/ unsplitted.jsonl dense
    # python3 scripts/mmimdb_as_graph.py ../dataset/mmimdb/ unsplitted.jsonl medium_weight
    save_name = sys.argv[3]
    dobj = format_mmimdb_add_staff(sys.argv[1], sys.argv[2], save_name)
    dobj = format_mmimdb_add_edge(dobj)

    dflatten, edges, idx2node = save_mmimdb_dataset_to_csv(dobj, sys.argv[1], save_name)
    print(dflatten[0])