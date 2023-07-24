import argparse
from pathlib import Path

from joblib import Parallel, delayed

from submission.scifact import prepare_scifact_sent_submission, prepare_scifact_doc_submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare SciFact predictions for submission in AllenNLP.")
    parser.add_argument("in_dir", type=str, help="Input prediction directory")
    parser.add_argument("out_dir", type=str, help="Directory to write submission files")
    parser.add_argument("-p", "--n-jobs", type=int, default=1, help="Number of workers to run")
    parser.add_argument("--sentence", action="store_true", help="Prepare for sentence model")
    parser.add_argument("--conversion-mode", type=str, choices=["meanproba", "majority", "hybrid"], help="Aggregation mode for sentence models. Choices: 'meanproba', 'majority', 'hybrid'. Ignored if '--sentence' flag is not used")
    
    args = parser.parse_args()
    
    indir = Path(args.in_dir)
    outdir = Path(args.out_dir)
    
    print("Start processing...")
    if args.sentence:
        _ = Parallel(n_jobs=args.n_jobs)(delayed(prepare_scifact_sent_submission)(p, outdir, args.conversion_mode) for p in indir.iterdir())
    else:
        _ = Parallel(n_jobs=args.n_jobs)(delayed(prepare_scifact_doc_submission)(p, outdir) for p in indir.iterdir())
    print("End processing...")
