import re
import argparse
from pathlib import Path
from collections import namedtuple

from tqdm import tqdm

from mod_eval_da import eval_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch predict on DA models.")
    parser.add_argument("doc_dir", type=str, help="Directory to document data")
    parser.add_argument("da_dir", type=str, help="Directory to DA models")
    parser.add_argument("out_dir", type=str, help="Directory to write prediction data")
    
    args = parser.parse_args()

    data_p = Path(args.doc_dir)
    # data_p = list(data_p.glob("*dev*")) + list(data_p.glob("*test*"))
    data_p = [p for p in data_p.iterdir() if p.is_file() and not p.stem.startswith("fever-climatefever")]
    if len(data_p) == 0:
        raise FileNotFoundError("No data files found in directory!")
    
    model_p = list(Path(args.da_dir).glob("*.tar.gz"))
    if len(model_p) == 0:
        raise FileNotFoundError("No DA models found! Please make sure it is in compressed '.tar.gz'.")

    out_p = Path(args.out_dir)
    out_p.mkdir(exist_ok=True)
    
    # <model-trained-on>-<model-type>.[dev|test|all].jsonl
    pred_file_fmt = "{0}-{1}.{2}.jsonl"
    
    for model in model_p:
        for data in data_p:
            data_fn = data.stem.split(".")
            model_fn = re.findall(r"(.*)-da-.*", model.stem)[0]
            args_proxy = namedtuple("args", ["archive_file", "in_file", "log", "cuda_device"])
            
            write_p = out_p / data_fn[0] / pred_file_fmt.format(model_fn, "da", data_fn[1])
            write_p.parent.mkdir(exist_ok=True)
            
            opts = args_proxy(str(model.resolve()), str(data.resolve()), str(write_p.resolve()), 0)
            eval_model(opts)
