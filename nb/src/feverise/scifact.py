from copy import deepcopy
import constants
from feverise.util import denormalise_title

def feverise_claims(data):
    """
    Convert SciFact Claims to FEVER format
    """
    sf_fever_label = {
        "SUPPORT": constants.LOOKUP["label"]["s"],
        "CONTRADICT": constants.LOOKUP["label"]["r"]
    }
    fdata = []
    for doc in data:
        fdoc = {
            "id": doc["id"], 
            "claim": doc["claim"], 
            "label": constants.LOOKUP["label"]["nei"], 
            "verifiable": constants.LOOKUP["verifiable"]["no"],
            "elab": None, # store labels for each evidence
            "evidence": [] # [[[Annotation ID, Evidence ID, Wikipedia URL, sentence ID], ...]]
        }
        if doc["evidence"]:
            label_ls, evi_ls = [], []
            for eid, es in doc["evidence"].items():
                for i in es:
                    for j in i["sentences"]:
                        label_ls.append(sf_fever_label[i["label"]])
                        evi_ls.append([None, None, str(eid), j])
            fdoc["elab"] = label_ls
            fdoc["evidence"].append(evi_ls)
            if sf_fever_label["CONTRADICT"] in label_ls:
                fdoc["label"] = constants.LOOKUP["label"]["r"]
                fdoc["verifiable"] = constants.LOOKUP["verifiable"]["yes"]
            elif sf_fever_label["SUPPORT"] in label_ls:
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
    fdata_with_title = []
    for doc in data:
        fdoc = {
            "id": str(doc["doc_id"]),
            "title": doc["title"],
            "structured": doc["structured"],
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
        
        fdata_with_title.append(fdoc)
        
    return fdata_with_title

def feverise_corpus_titleid(feverise_data):
    """
    Rearrange Feverised SciFact corpus to use title as doc_id.
    Required for pipelines and fully FEVER format compliant.
    """
    sf_titleid = []
    for doc in feverise_data:
        fdoc = deepcopy(doc)
        fdoc["id"] = denormalise_title(doc["title"])
        fdoc["original_id"] = doc["id"]
        del fdoc["title"]
        sf_titleid.append(fdoc)
    
    return sf_titleid
