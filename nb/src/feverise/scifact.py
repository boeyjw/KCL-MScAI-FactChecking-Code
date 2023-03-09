from . import constants

def feverise_claims(data):
    """
    Convert SciFact Claims to FEVER format
    """
    fdata = []
    for doc in data:
        fdoc = {
            "id": doc["id"], 
            "claim": doc["claim"], 
            "label": constants.LOOKUP["label"]["nei"], 
            "verifiable": constants.LOOKUP["verifiable"]["no"],
            "evidence": [] # [[[Annotation ID, Evidence ID, Wikipedia URL, sentence ID], ...]]
        }
        if doc["evidence"]:
            label_ls, evi_ls = [], []
            for eid, es in doc["evidence"].items():
                for i in es:
                    label_ls.append(i["label"])
                    for j in i["sentences"]:
                        evi_ls.append([None, None, str(eid), j])
            fdoc["evidence"].append(evi_ls)
            if "CONTRADICT" in label_ls:
                fdoc["label"] = constants.LOOKUP["label"]["r"]
                fdoc["verifiable"] = constants.LOOKUP["verifiable"]["yes"]
            elif "SUPPORT" in label_ls:
                fdoc["label"] = constants.LOOKUP["label"]["s"]
                fdoc["verifiable"] = constants.LOOKUP["verifiable"]["yes"]
            # label only has 'NO INFO', keep default doc with evidence
        else:
            fdoc["evidence"].append([[None, None, None, None]])
        fdata.append(fdoc)
    return fdata

def feverise_corpus(data):
    """
    Convert SciFact Corpus to FEVER format
    """
    fdata = []
    for doc in data:
        fdoc = {
            "id": str(doc["doc_id"]),
            "text": None,
            "lines": None
        }
        tmp_l, tmp_t = "", ""
        for i, line in enumerate(doc["abstract"]):
            # No tags, possible WIP?
            tmp_t += f"{line.strip()} "
            tmp_l += f"{i}\t{line.strip()}\n"
        fdoc["text"] = tmp_t.strip()
        fdoc["lines"] = tmp_l.strip()
        fdata.append(fdoc)
    return fdata
