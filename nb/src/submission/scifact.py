from pathlib import Path
from collections import Counter

import pandas as pd

import constants
from gen.util import read_data, write_jsonl
from rte.aggregate import agg_predict, agg_predict_proba

LABEL2LABEL = {"SUPPORTS": "SUPPORT", "REFUTES": "CONTRADICT"}

def _simple_convert(predictions):
    submission = []
    
    for doc in predictions:
        sub_doc = {
            "id": int(doc["claim_id"].split("|")[1]),
            "evidence": {}
        }
        if doc["predicted_label"] != constants.LOOKUP["label"]["nei"]:
            for evidence in doc["predicted_evidence"]:
                page = str(evidence[0])
                line = int(evidence[1])
                if page in sub_doc["evidence"]:
                    sub_doc["evidence"][page]["sentences"].append(line)
                    sub_doc["evidence"][page]["sentences"] = sorted(sub_doc["evidence"][page]["sentences"])
                else:
                    sub_doc["evidence"][page] = {
                        "sentences": [line],
                        "label": LABEL2LABEL[doc["predicted_label"]]
                    }
        submission.append(sub_doc)
        
    return submission

def _hybrid_convert(predictions):
    staging = {}
    
    for ind, doc in predictions.groupby(["id", "page"], sort=False).agg({"plabel": list, "predicted_proba": list, "line": list}).iterrows():
        if ind[0] not in staging:
            staging[ind[0]] = {"evidence": {}}
        cnt = Counter(doc["plabel"])
        if len(cnt) > 1:
            keep_ev, keep_lab, probas = [], [], []
            for ev, plab, proba in zip(doc["line"], doc["plabel"], doc["predicted_proba"]):
                if constants.ID2LABEL[plab] != constants.LOOKUP["label"]["nei"]:
                    keep_ev.append(ev)
                    keep_lab.append(plab)
                    probas.append(proba)
            cnt = Counter(keep_lab).most_common(2)
            if len(cnt) > 1:
                if cnt[0][1] == cnt[1][1]:
                    # mean probability if S == R
                    label = constants.ID2LABEL[agg_predict_proba(probas)]
                else:
                    # majority if unequal
                    label = constants.ID2LABEL[cnt[0][0]]
            else:
                label = constants.ID2LABEL[keep_lab[0]]
            staging[ind[0]]["evidence"][ind[1]] = {
                "sentences": sorted(keep_ev),
                "label": LABEL2LABEL[label]
            }
        else:
            label = constants.ID2LABEL[doc["plabel"][0]]
            if label != constants.LOOKUP["label"]["nei"]:
                staging[ind[0]]["evidence"][ind[1]] = {
                    "sentences": sorted(doc["line"]),
                    "label": LABEL2LABEL[label]
                }
    
    # convert to jsonl
    submissions = [{"id": k, "evidence": v["evidence"]} for k, v in staging.items()]
    
    return submissions

def prepare_scifact_doc_submission(pred_p: Path, outp: Path):
    predictions = read_data(pred_p)
    submissions = _simple_convert(predictions)
    
    return write_jsonl(outp / (pred_p.stem.split(".")[0] + ".test.submission.jsonl"), submissions)

def prepare_scifact_sent_submission(pred_p: Path, outp: Path, conversion_mode: str):
    predictions = pd.DataFrame(read_data(pred_p))
    predictions["plabel"] = predictions["predicted_label"].map(constants.LABEL2ID)
    cmode = conversion_mode.lower().strip()
    
    # convert
    submissions = None
    if cmode in ["majority", "meanproba"]:
        # aggregate
        pred_grp = predictions.groupby("claim_id", sort=False, as_index=False)
        if cmode == "majority":
            pred_grp = pred_grp.agg({
                "plabel": agg_predict,
                "predicted_evidence": list
            }).rename(columns={"plabel": "predicted_label"})
        elif cmode == "meanproba":
            pred_grp = pred_grp.agg({
                "predicted_proba": agg_predict_proba,
                "predicted_evidence": list
            }).rename(columns={"predicted_proba": "predicted_label"})
        else:
            raise NotImplementedError("Only 'majority' and 'meanproba' aggregations were implemented.")
        pred_grp["predicted_label"] = pred_grp["predicted_label"].map(constants.ID2LABEL)
        submissions = _simple_convert(pred_grp.to_dict("records"))
    elif cmode == "hybrid":
        # mixed evidence labelling
        predictions["id"] = predictions["claim_id"].apply(lambda x: int(x.split("|")[1]))
        predictions["page"] = predictions["predicted_evidence"].apply(lambda x: str(x[0]))
        predictions["line"] = predictions["predicted_evidence"].apply(lambda x: int(x[1]))
        predictions = predictions.set_index("id")
        submissions = _hybrid_convert(predictions)
    
    return write_jsonl(outp / (pred_p.stem.split(".")[0] + ".test.submission.jsonl"), submissions)
