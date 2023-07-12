import sys
sys.path.insert(0, "/users/k21190024/study/fact-checking-repos/fever/baseline/src")
import re
from pathlib import Path
from collections import namedtuple

from tqdm import tqdm

from scripts.rte.da.boey_eval_da import eval_model


if __name__ == "__main__":
    data_p = Path("/users/k21190024/study/fact-checking-repos/fever/baseline/dumps/bert-data-doc-evidence")
    data_p = list(data_p.glob("*dev*")) + list(data_p.glob("*test*"))
    data_p = [p for p in data_p if "fever-climatefever" not in p.stem]
    
    model_p = Path("/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/models/decomposable-attention").glob("*.out")
    out_p = Path("/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/predictions/doc")
    # <model-trained-on>-<model-type>.[dev|test].jsonl
    pred_file_fmt = "{0}-{1}.{2}.jsonl"
    
    for model in model_p:
        for data in data_p:
            data_fn = data.stem.split(".")
            model_fn = re.findall(r"(.*)-da-.*", model.stem)[0]
            args_proxy = namedtuple("args", ["archive_file", "in_file", "log", "cuda_device"])
            opts = args_proxy(str(model / "model.tar.gz"), str(data), str(out_p / data_fn[0] / pred_file_fmt.format(model_fn, "da", data_fn[1])), 0)
            eval_model(opts)
