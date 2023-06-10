import sqlite3
from copy import deepcopy
from collections import defaultdict

def denormalise_title(s):
    return s.replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace(":", "-COLON-").strip(".")

def count_evidences(claims):
    n_evidences = defaultdict(list)
    for d in claims:
        n_evidences[d["label"]].append(0)
        for i in d["evidence"]:
            n_evidences[d["label"]][-1] += len(i)
        if "other_evidence" in d and d["other_evidence"] is not None:
            for i in d["other_evidence"]:
                n_evidences[d["label"]][-1] += len(i)
    
    return n_evidences

def replace_id_with_titleid(db_path, doc):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    evidences = doc["evidence"]
    titleid_evidences = []
    for e in evidences:
        if e[0][2] is not None:
            cur.execute("SELECT id FROM documents WHERE original_id = ?", (e[0][2],))
            res = cur.fetchone()[0]
            titleid_evidences.append([[e[0][0], e[0][1], res, e[0][3]]])
        else:
            titleid_evidences.append([[None]*4])
    new_doc = deepcopy(doc)
    new_doc["evidence"] = titleid_evidences
    
    if "other_evidence" in doc:
        if doc["other_evidence"] is not None:
            other_evidences = doc["other_evidence"]
            other_titleid_evidences = []
            for e in other_evidences:
                if e[0][2] is not None:
                    cur.execute("SELECT id FROM documents WHERE original_id = ?", (e[0][2],))
                    res = cur.fetchone()[0]
                    other_titleid_evidences.append([[e[0][0], e[0][1], res, e[0][3]]])
                else:
                    other_titleid_evidences.append(e[0])
        else:
            other_titleid_evidences = doc["other_evidence"]
        new_doc["other_evidence"] = other_titleid_evidences
    
    conn.close()
    
    return new_doc
