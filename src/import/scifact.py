def scifact_preproc(doc, corpus_col):
    """
    Preprocess SciFact claims to include support into each claim document for ease of processing later
    Usage:
        scifact_data = db["scifact_data"]
        scifact_data.drop()
        sf_data_proc = Parallel(2, backend="threading", verbose=3)(delayed(scifact_preproc)(doc, scifact_corpus) for doc in sf_data)
        sf_data_col = scifact_data.insert_many(sf_data_proc)
    """
    if "evidence" not in doc:
        return doc
    for iind, i in enumerate(doc["cited_doc_ids"]):
        cite = corpus_col.find_one({"doc_id": i})
        if doc["evidence"] and str(i) in doc["evidence"]:
            for jind, j in enumerate(doc["evidence"][str(i)]):
                sentences = [cite["abstract"][k] for k in j["sentences"]]
                doc["evidence"][str(i)][jind]["sentences"] = sentences
        del cite["_id"]
        doc["cited_doc_ids"][iind] = cite
    return doc