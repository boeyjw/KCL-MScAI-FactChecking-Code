import gzip
import json

def read_gzip_data(fp):
    with gzip.GzipFile(fp, "r") as gzfn:
        data = [json.loads(l.decode("utf8")) for l in gzfn.readlines()]
    return data

def write_gzip_data(fp, payload):
    with gzip.GzipFile(fp, "w") as gzfn:
        gzfn.write(json.dumps(payload).encode("utf-8"))
    return fp