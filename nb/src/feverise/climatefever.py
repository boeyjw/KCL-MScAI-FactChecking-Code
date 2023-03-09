from . import constants

def feverise_climatefever(data) -> tuple:
    """
    Feverise climate-FEVER claims and corpus
    """
    claims_ls, corpus_d = [], {}
    for doc in data:
        cdoc = {
            "id": int(doc["claim_id"]),
            "claim": doc["claim"],
            "elab": None,  # evidence labels
            "evidence": []  # evidences
        }
        if doc["claim_label"] == "NOT_ENOUGH_INFO":
            cdoc["verifiable"] = constants.LOOKUP["verifiable"]["no"]
            cdoc["label"] = constants.LOOKUP["label"]["nei"]
        else:
            cdoc["verifiable"] = constants.LOOKUP["verifiable"]["yes"]
            cdoc["label"] = constants.LOOKUP["label"]["s"] if doc["claim_label"] == "SUPPORTS" else constants.LOOKUP["label"]["r"]

        evidence_ls, label_ls = [], []
        for evidence in doc["evidences"]:
            eid = evidence["evidence_id"].split(":")  # id, sentence_id
            # claims section
            evidence_ls.append([None, None, eid[0], eid[1]])
            label_ls.append(evidence["evidence_label"])
            # corpus section
            if eid[0] in corpus_d and eid[1] not in corpus_d[eid[0]]:
                corpus_d[eid[0]][eid[1]] = evidence["evidence"]
            else:
                corpus_d[eid[0]] = {eid[1]: evidence["evidence"]}
        # append claim
        cdoc["elab"] = label_ls
        cdoc["evidence"].append(evidence_ls)
        claims_ls.append(cdoc)

    # process corpus
    corpus_ls = []
    for cid, sentences in corpus_d.items():
        cdoc = {
            "id": cid,
            "text": "",
            "lines": []
        }
        for sid, l in sentences.items():
            cdoc["text"] += l + " "
            cdoc["lines"].append(f"{sid}\t{l}")
        cdoc["lines"] = "\n".join(cdoc["lines"])
        cdoc["text"].strip()
        cdoc["lines"].strip()
        corpus_ls.append(cdoc)
    return claims_ls, corpus_ls