from copy import deepcopy
import constants
from feverise.util import denormalise_title

def _feverise_corpus(corpus_d):
    """
    Process corpus
    """
    corpus_ls = []
    for cid, sentences in corpus_d.items():
        cdoc = {
            "id": cid,
            "text": "",
            "lines": []
        }
        for sid, l in sorted(sentences.items()):
            cdoc["text"] += l + " "
            cdoc["lines"].append(f"{sid}\t{l}")
        cdoc["lines"] = "\n".join(cdoc["lines"])
        
        cdoc["text"].strip()
        cdoc["lines"].strip()
        corpus_ls.append(cdoc)
    return corpus_ls

def feverise_corpus_titleid(feverise_data):
    """
    Rearrange Feverised Climate-FEVER corpus to use title as doc_id.
    Required for pipelines and fully FEVER format compliant.
    """
    cf_titleid = []
    for doc in feverise_data:
        cdoc = deepcopy(doc)
        cdoc["id"] = denormalise_title(doc["id"])
        cdoc["original_id"] = doc["id"]
        cf_titleid.append(cdoc)
    
    return cf_titleid

def _init_assumed_claim(doc):
    """
    Assume DISPUTED claims are NOT ENOUGH INFO
    rationale: if a claim is disputed, we can assume that there is
    insufficient evidence to assert the claim as annotators have
    different expert opinion regarding the matter
    """
    cdoc = {
        "id": int(doc["claim_id"]),
        "claim": doc["claim"],
        "label": None,
        "elab": None,  # evidence labels
        "is_disputed": False,
        "evidence": [],  # evidences
        "other_elab": None,  # NEI claim label evidence labels
        "other_evidence": []  # NEI claim label evidences
    }
    if doc["claim_label"] == "NOT_ENOUGH_INFO":
        cdoc["verifiable"] = constants.LOOKUP["verifiable"]["no"]
        cdoc["label"] = constants.LOOKUP["label"]["nei"]
    elif doc["claim_label"] == "DISPUTED":
        cdoc["verifiable"] = constants.LOOKUP["verifiable"]["no"]
        cdoc["label"] = constants.LOOKUP["label"]["nei"]
        cdoc["is_disputed"] = True
    else:
        cdoc["verifiable"] = constants.LOOKUP["verifiable"]["yes"]
        cdoc["label"] = constants.LOOKUP["label"]["s"] if doc["claim_label"] == "SUPPORTS" else constants.LOOKUP["label"]["r"]
    
    return cdoc
    
def _init_paper_claim(doc):
    """
    Exclude DISPUTED claims identical to the paper methodology
    for evaluating on FEVER methodology
    """
    cdoc = {
        "id": int(doc["claim_id"]),
        "claim": doc["claim"],
        "label": None,
        "elab": None,  # evidence labels
        "evidence": [],  # evidences
        "other_elab": None,  # NEI claim label evidence labels
        "other_evidence": []  # NEI claim label evidences
    }
    if doc["claim_label"] == "NOT_ENOUGH_INFO":
        cdoc["verifiable"] = constants.LOOKUP["verifiable"]["no"]
        cdoc["label"] = constants.LOOKUP["label"]["nei"]
    elif doc["claim_label"] == "DISPUTED":
        return None
    else:
        cdoc["verifiable"] = constants.LOOKUP["verifiable"]["yes"]
        cdoc["label"] = constants.LOOKUP["label"]["s"] if doc["claim_label"] == "SUPPORTS" else constants.LOOKUP["label"]["r"]
    
    return cdoc

def feverise_climatefever(data) -> tuple:
    """
    Feverise climate-FEVER claims and corpus
    
    Returns paper FEVER compliant claim-evidence pair (exclude DISPUTED claims), 
    assumed claim-evidence pair where DISPUTED claims are treated as NOT ENOUGH INFO
    and corpus with ordered lines
    """
    def _add_evidences_to_claim(cdoc, doc, evidence_ls, label_ls, other_evidence_ls, other_label_ls):
        if cdoc is None:
            return
        # add evidence and evidence labels
        if doc["claim_label"] in ["NOT_ENOUGH_INFO", "DISPUTED"]:
            # FEVER claims with NOT ENOUGH INFO do not have evidences
            cdoc["evidence"].append([[None]*4])
            if doc["claim_label"] == "DISPUTED" and "is_disputed" in cdoc:
                cdoc["is_disputed"] = True
        else:
            cdoc["evidence"].append(evidence_ls)
            cdoc["elab"] = label_ls

        if other_evidence_ls:
            cdoc["other_evidence"].append(other_evidence_ls)
            cdoc["other_elab"] = other_label_ls
            
    cf_fever_labels = {
        "NOT_ENOUGH_INFO": constants.LOOKUP["label"]["nei"],
        "SUPPORTS": constants.LOOKUP["label"]["s"],
        "REFUTES": constants.LOOKUP["label"]["r"],
        "DISPUTED": constants.LOOKUP["label"]["nei"]
    }
    
    assumed_ls, paper_ls = [], []
    corpus_d = {}
    for doc in data:
        evidence_ls, label_ls = [], []
        other_evidence_ls, other_label_ls = [], []
        for evidence in doc["evidences"]:
            eid_tokens = evidence["evidence_id"].split(":")  # [id, sentence_id]
            eid = "".join(eid_tokens[0:-1])  # evidence key
            sid = int(eid_tokens[-1])  # sentence index
            # claims section
            if doc["claim_label"] not in ["NOT_ENOUGH_INFO", "DISPUTED"] and evidence["evidence_label"] == doc["claim_label"]:
                evidence_ls.append([None, None, eid, sid])
                label_ls.append(cf_fever_labels[evidence["evidence_label"]])
            else:
                # keep evidence and repective labels for evaluation purposes
                other_evidence_ls.append([None, None, eid, sid])
                other_label_ls.append(cf_fever_labels[evidence["evidence_label"]])
            # corpus section
            if eid in corpus_d:
                if sid not in corpus_d[eid]:
                    corpus_d[eid][sid] = evidence["evidence"]
            else:
                corpus_d[eid] = {sid: evidence["evidence"]}
                
        cdoc_assumed = _init_assumed_claim(doc)
        cdoc_paper = _init_paper_claim(doc)
        
        _add_evidences_to_claim(cdoc_assumed, doc, evidence_ls, label_ls, other_evidence_ls, other_label_ls)
        _add_evidences_to_claim(cdoc_paper, doc, evidence_ls, label_ls, other_evidence_ls, other_label_ls)
        
        assumed_ls.append(cdoc_assumed)
        if cdoc_paper is not None:
            paper_ls.append(cdoc_paper)
            
    # process corpus
    corpus_ls = _feverise_corpus(corpus_d)
    
    return paper_ls, assumed_ls, corpus_ls