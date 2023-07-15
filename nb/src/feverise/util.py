import sqlite3
from copy import deepcopy
from collections import defaultdict

def replace_id_with_titleid(db_path, doc):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    evidences = doc["evidence"]
    titleid_evidences = []
    for evidence in evidences:
        ev_ls = []
        for ev in evidence:
            if ev[2] is not None:
                cur.execute("SELECT id FROM documents WHERE original_id = ?", (ev[2],))
                res = cur.fetchone()[0]
                ev_ls.append([ev[0], ev[1], res, ev[3]])
            else:
                ev_ls.append([None]*4)
        titleid_evidences.append(ev_ls)
    new_doc = deepcopy(doc)
    new_doc["evidence"] = titleid_evidences
    
    if "other_evidence" in doc:
        if doc["other_evidence"] is not None:
            other_evidences = doc["other_evidence"]
            other_titleid_evidences = []
            for evidence in other_evidences:
                ev_ls = []
                for ev in evidence:
                    if ev[2] is not None:
                        cur.execute("SELECT id FROM documents WHERE original_id = ?", (ev[2],))
                        res = cur.fetchone()[0]
                        ev_ls.append([ev[0], ev[1], res, ev[3]])
                    else:
                        ev_ls.append([None]*4)
                other_titleid_evidences.append(ev_ls)
        else:
            other_titleid_evidences = doc["other_evidence"]
        new_doc["other_evidence"] = other_titleid_evidences
    
    conn.close()
    
    return new_doc
