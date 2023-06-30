import gzip
import json
import pickle as pkl
import os
import csv
import unicodedata

def is_json_suffix(fp) -> str:
    fname = os.path.basename(fp)
    return ".json" in fname or ".jsonl" in fname

def read_data(fp):
    fn = gzip.open(fp, "rb") if fp.suffix == ".gz" else open(fp, "r")
    fp_str = str(fp)
    try:
        if ".jsonl" in os.path.basename(fp_str):
            data = [json.loads(l.decode("utf8") if fp.suffix == ".gz" else l) for l in fn.readlines()]
        elif ".json" in os.path.basename(fp_str):
            data = json.loads(fn.read())
        elif ".pkl.gz" in os.path.basename(fp_str):
            data = pkl.loads(fn.read())
        else:
            raise NotImplementedError(f"{os.path.basename(fp)} suffix is unsupported.")
    finally:
        fn.close()
    return data

def write_gzip_data(fp, payload):
    fp = str(fp)
    with gzip.open(fp, "wb") as gzfn:
        if is_json_suffix(fp):
            gzfn.write(json.dumps(payload).encode("utf-8"))
        elif ".pkl.gz" in os.path.basename(fp):
            gzfn.write(pkl.dumps(payload))
    return fp

def write_jsonl(fp, payload):
    with open(fp, "w") as fn:
        for doc in payload:
            fn.writelines(json.dumps(doc, ensure_ascii=False) + "\n")
    return fp

def normalize(text):
    """
    Resolve different type of unicode encodings.
    
    Reference: https://github.com/facebookresearch/DrQA/blob/main/drqa/retriever/utils.py
    """
    return unicodedata.normalize('NFD', text)

def denormalise_title(s):
    return s.replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace(":", "-COLON-").strip(".")

def normalise_title(s):
    return s.replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")").replace("-COLON-", ":").strip(".")