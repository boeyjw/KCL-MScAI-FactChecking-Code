import sqlite3
from copy import deepcopy
from collections import defaultdict

def denormalise_title(s):
    return s.replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace(":", "-COLON-")

def count_evidences(claims):
    n_evidences = defaultdict(list)
    for d in claims:
        n_evidences[d["label"]].append(0)
        for i in d["evidence"]:
            n_evidences[d["label"]][-1] += len(i)
        if "other_evidence" in d:
            for i in d["other_evidence"]:
                n_evidences[d["label"]][-1] += len(i)
    
    return n_evidences

def replace_id_with_titleid(db_path, doc):
    if doc["label"] == "NOT ENOUGH INFO":
        return doc
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    evidences = doc["evidence"]
    titleid_evidences = []
    for e in evidences[0]:
        cur.execute("SELECT id FROM documents WHERE original_id = ?", (e[2],))
        res = cur.fetchone()[0]
        titleid_evidences.append([e[0], e[1], res, e[3]])
    new_doc = deepcopy(doc)
    new_doc["evidence"] = []
    new_doc["evidence"].append(titleid_evidences)
    
    other_evidences = doc["other_evidence"]
    if other_evidences:
        other_titleid_evidences = []
        for e in other_evidences[0]:
            cur.execute("SELECT id FROM documents WHERE original_id = ?", (e[2],))
            res = cur.fetchone()[0]
            other_titleid_evidences.append([e[0], e[1], res, e[3]])
        new_doc["other_evidence"] = []
        new_doc["other_evidence"].append(other_titleid_evidences)
    
    conn.close()
    
    return new_doc
