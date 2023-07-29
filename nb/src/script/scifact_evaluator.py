import subprocess
import shlex
import sys
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    # Reference: https://github.com/allenai/scifact-evaluator/tree/master
    script_p = "/users/k21190024/study/fact-check-transfer-learning/repos/scifact/scifact-evaluator/evaluator/eval.py"
    
    parser = ArgumentParser(description="SciFact evaluation.")
    parser.add_argument("labels_file", type=str, help="File with gold evidence.")
    parser.add_argument("preds_dir", type=str, help="File with predictions.")
    parser.add_argument("metrics_output_dir",
                        type=str,
                        help="Location of output metrics file",
                        default="metrics.json")
    
    args = parser.parse_args()
    
    indir = Path(args.preds_dir)
    outdir = Path(args.metrics_output_dir)
    
    for p in tqdm(indir.iterdir()):
        _ = subprocess.run(shlex.split(f"""
            {sys.executable} {script_p} 
            --labels_file {args.labels_file} 
            --preds_file {p} 
            --metrics_output_file {outdir / (p.stem + ".jsonl")} 
            --verbose
        """))