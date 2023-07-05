import argparse
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder

import constants
from retrieval.fever_doc_db import FeverDocDB
from gen import util

def prepare_doc(doc, db_path, sentence_pair: bool, cross_encoder_name: str, max_evidence: int, nei_max_evidence: int):
    db = FeverDocDB(db_path)
    sentences = []
    sent_line = []
    for ev in doc["evidence"]:
        e = ev[0]
        if e[2] is not None:
            # handle duplicate evidence
            sl = e[2] + str(e[3])
            if sl in sent_line:
                continue
            else:
                sent_line.append(sl)
            lines = db.get_doc_lines(util.denormalise_title(e[2]))
            if lines is not None:
                for line in lines.split("\n"):
                    l = line.split("\t")
                    # handle dirty evidence lines in db
                    if l[0].isdigit() and l[1].strip() and int(l[0]) == int(e[3]):
                        sentences.append(util.normalise_title(l[1]))
    
    # limit number of evidences and retrieve only closest evidence to prevent major skew in target distribution
    if len(sentences) > max_evidence:
        ce = CrossEncoder(cross_encoder_name)
        score = ce.predict([[doc["claim"], s] for s in sentences])
        score = sorted(list(zip(sentences, score)), reverse=True)
        sentences = [s[0] for s in score[:max_evidence]]
    if doc["label"] == constants.LOOKUP["label"]["nei"] and len(sentences) > nei_max_evidence:
        # assume already sorted since using nei_sampling.py script which is already sorted
        sentences = sentences[:nei_max_evidence]
    
    if sentence_pair:
        return [{"claim": doc["claim"], "evidence": s, "labels": constants.LABEL2ID[doc["label"]]} for s in sentences]
    else:
        return {"claim": doc["claim"], "evidence": " ".join(sentences), "labels": constants.LABEL2ID[doc["label"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert dataset into claim-evidence-label triplets use for transformer library.")
    parser.add_argument("in_file", type=str, help="Input claim data path")
    parser.add_argument("out_file", type=str, help="Path to write data")
    parser.add_argument("db_path", type=str, help="Path to corpus index database")
    parser.add_argument("-p", "--n-jobs", type=int, default=1, help="Number of workers to run")
    parser.add_argument("--sentence-pair", action="store_true", help="Return as claim-evidence sentence pair")
    parser.add_argument("-c", "--cross-encoder-name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="SentenceTransformer Cross-Encoder to use for sentence retrieval ranking.")
    parser.add_argument("--max-evidence", type=int, default=99, help="Maximum number of evidence to use")
    parser.add_argument("--nei-max-evidence", type=int, default=99, help="Maximum number of NEI claim-evidence to use")
    
    args = parser.parse_args()
    
    infile = Path(args.in_file)
    outfile = Path(args.out_file)
    
    print("Start processing...")
    datain = util.read_data(infile)
    res = Parallel(n_jobs=args.n_jobs)(delayed(prepare_doc)(doc, args.db_path, args.sentence_pair, args.cross_encoder_name, args.max_evidence, args.nei_max_evidence) for doc in datain)
    
    post_proc = []
    if args.sentence_pair:
        for r in tqdm(res):
            post_proc += r
    print("End processing...")
    util.write_jsonl(args.out_file, post_proc if args.sentence_pair else res)