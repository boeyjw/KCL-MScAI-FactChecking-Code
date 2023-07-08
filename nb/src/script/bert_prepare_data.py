import argparse
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder

import constants
from retrieval.fever_doc_db import FeverDocDB
from gen import util

def prepare_doc(doc, db_path, sentence_pair: bool, cross_encoder_name: str, max_evidence: int, nei_max_evidence: int, id_prefix: str, pipeline_mode: bool):
    db = FeverDocDB(db_path)
    sentences = []
    elab = []
    sent_line = []
    predicted_evidence = []  # only use in pipeline mode
    for i, ev in enumerate(doc["predicted_evidence"] if pipeline_mode else doc["evidence"]):
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
                        predicted_evidence.append([e[2], e[3]])
                        if not pipeline_mode and "elab" in doc:
                            elab.append(doc["elab"][i] if doc["elab"] else doc["label"])
    
    # limit number of evidences and retrieve only closest evidence to prevent major skew in target distribution
    if len(sentences) > max_evidence:
        ce = CrossEncoder(cross_encoder_name)
        score = ce.predict([[doc["claim"], s] for s in sentences])
        score = sorted(list(zip(score, sentences, predicted_evidence)), key=lambda x: x[0], reverse=True)
        sentences = [s[1] for s in score[:max_evidence]]
        predicted_evidence = [s[2] for s in score[:max_evidence]]
    if not pipeline_mode and doc["label"] == constants.LOOKUP["label"]["nei"] and len(sentences) > nei_max_evidence:
        # assume already sorted since using nei_sampling.py script which is already sorted
        sentences = sentences[:nei_max_evidence]
    
    if sentence_pair:
        if not pipeline_mode and "elab" in doc:
            return [
                {
                    "claim": doc["claim"], 
                    "evidence": s, 
                    "labels": constants.LABEL2ID[el], 
                    "claim_id": f"{id_prefix}|{doc['id']}"
                } for el, s in zip(elab, sentences)
            ]
        else:
            return [
                {
                    "claim": doc["claim"], 
                    "evidence": s, 
                    "labels": None if pipeline_mode else constants.LABEL2ID[doc["label"]], 
                    "predicted_evidence": pe if pipeline_mode else None,
                    "claim_id": f"{id_prefix}|{doc['id']}"
                } for s, pe in zip(sentences, predicted_evidence)
            ]
    else:
        return {
            "claim": doc["claim"], 
            "evidence": " ".join(sentences), 
            "labels": None if pipeline_mode else constants.LABEL2ID[doc["label"]], 
            "predicted_evidence": predicted_evidence if pipeline_mode else None,
            "claim_id": f"{id_prefix}|{doc['id']}"
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert dataset into claim-evidence-label triplets use for transformer library.")
    parser.add_argument("in_file", type=str, help="Input claim data path")
    parser.add_argument("out_file", type=str, help="Path to write data")
    parser.add_argument("db_path", type=str, help="Path to corpus index database")
    parser.add_argument("-p", "--n-jobs", type=int, default=1, help="Number of workers to run")
    parser.add_argument("--sentence-pair", action="store_true", help="Return as claim-evidence sentence pair")
    parser.add_argument("-c", "--cross-encoder-name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="SentenceTransformer Cross-Encoder to use for sentence retrieval ranking.")
    parser.add_argument("--max-evidence", type=int, default=99999, help="Maximum number of evidence to use")
    parser.add_argument("--nei-max-evidence", type=int, default=99999, help="Maximum number of NEI claim-evidence to use")
    parser.add_argument("--pipeline-mode", action="store_true", help="Retrieve sentences for 'predicted_evidence' instead of 'evidence'")
    
    args = parser.parse_args()
    
    infile = Path(args.in_file)
    outfile = Path(args.out_file)
    
    print("Start processing...")
    datain = util.read_data(infile)
    res = Parallel(n_jobs=args.n_jobs)(delayed(prepare_doc)(
        doc, args.db_path, args.sentence_pair, args.cross_encoder_name, 
        args.max_evidence, args.nei_max_evidence, infile.parent.stem.split("-")[0], 
        args.pipeline_mode
    ) for doc in datain)
    
    post_proc = []
    if args.sentence_pair:
        for r in tqdm(res):
            post_proc += r
    print("End processing...")
    util.write_jsonl(args.out_file, post_proc if args.sentence_pair else res)