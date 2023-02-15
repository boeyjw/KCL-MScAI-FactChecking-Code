import gzip
import json
import pickle as pkl
import os

def is_json_suffix(fp):
    fname = os.path.basename(fp)
    return ".json" in fname or ".jsonl" in fname

def read_gzip_data(fp):
    fp = str(fp)
    with gzip.open(fp, "rb") as gzfn:
        if is_json_suffix(fp):
            try:
                data = json.loads(gzfn.read())
            except ValueError:
                data = [json.loads(l.decode("utf8")) for l in gzfn.readlines()]
        elif ".pkl.gz" in os.path.basename(fp):
            data = pkl.loads(gzfn.read())
        else:
            raise NotImplementedError(f"{os.path.basename(fp)} suffix is unsupported.")
    return data if len(data) > 1 else data[0]

def write_gzip_data(fp, payload):
    fp = str(fp)
    with gzip.open(fp, "wb") as gzfn:
        if is_json_suffix(fp):
            gzfn.write(json.dumps(payload).encode("utf-8"))
        elif ".pkl.gz" in os.path.basename(fp):
            gzfn.write(pkl.dumps(payload))
    return fp