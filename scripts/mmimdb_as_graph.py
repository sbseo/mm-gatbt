from collections import defaultdict
import sys
import os
import json
import pandas as pd


def format_mmimdb_add_staff(dataset_root_path):
    """return dictionary which includes movie info and staff.
    
    Requirement:
        `train.jsonl` created by mmimdb.py

    Args:
        dataset_root_path (str): root path mm-imdb dataset

    Returns:
        dobj: dictinary object where key is movie id and value is dictionary
    """

    features = ['director', 'producer', 'writer', 'cinematographer', 'art director']
    dobj = defaultdict()
    f = open(os.path.join(dataset_root_path, "dev.jsonl"), 'r')
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
        for idx2, d2 in dic.items():
            if idx == idx2:
                continue
            if set.intersection(staff, set(d2["staff"])):
                connected.append(idx2)
        edges = {"edges":connected}
        dic[idx].update(edges)

    return dic

def save_mmimdb_dataset_to_csv(dic, dataset_root_path):
    """Save to csv

    Args:
        dic (dict): {id:{id, feat, edges, lbl}}

    Returns:
        node_data.csv: id | feat | lbl
        edge_data.csv: src | dest
    """
    # for each dict, save it to 
    # key: {id, label, nodes}

    dic_flatten = defaultdict()
    edges = list()
    features = ["id", "text", "image", "label"]
    for idx, d in dic.items():
        dic_flatten[idx] = [d[feat] for feat in features]
        edges += [{"src": idx, "dst": dst} for dst in d["edges"]]

    node_data = pd.DataFrame.from_dict(dic_flatten, orient='index')
    node_data.to_csv(os.path.join(dataset_root_path, "node_data.csv"), index=False, header=features)

    edge_data = pd.DataFrame.from_records(edges)
    edge_data.to_csv(os.path.join(dataset_root_path, "edge_data.csv"), index=False)

    return dic_flatten, edges


if __name__=="__main__":
    # python3 scripts/add_connection.py ../dataset/mmimdb/
    dobj = format_mmimdb_add_staff(sys.argv[1])
    dobj = format_mmimdb_add_edge(dobj)
    dflatten, edges = save_mmimdb_dataset_to_csv(dobj, sys.argv[1])
    print(dflatten["1277953"])


