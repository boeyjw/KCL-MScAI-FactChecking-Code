# Sample NEI using BM25 + Cross-Encoder [cross-encoder/ms-marco-MiniLM-L-6-v2]
# https://github.com/beir-cellar/beir/tree/main/examples/retrieval/evaluation/reranking shows it has good performance to
# result

import argparse
from pathlib import Path
from multiprocessing import Pool
from copy import deepcopy

from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

import constants
from retrieval.fever_doc_db import FeverDocDB
from retrieval.retrieval import BM25DocumentRetriever
from gen import util


def run_sentence_retrieval(doc, db, ce):
    ce_input = []
    sents_idx = []
    page_ls = []
    for page in doc["predicted_pages"]:
        lines = db.get_doc_lines(util.denormalise_title(page))
        for line in lines.split("\n"):
            elem = line.split("\t")
            if elem[0].isdigit() and elem[1].strip():
                sents_idx.append(elem[0])
                page_ls.append(page)
                ce_input.append([doc["claim"], elem[1]])
    scores = ce.predict(ce_input)
    sents = sorted(list(zip(page_ls, sents_idx, scores)), key=lambda x: x[2], reverse=True)
    
    return doc["id"], [[None, None, s[0], int(s[1])] for s in sents]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evidence sampling using BM25 + Cross-Encoder")
    parser.add_argument("in_file", type=str, help="Input claim data path")
    parser.add_argument("out_file", type=str, help="Path to write sampled NEI data")
    parser.add_argument("db_path", type=str, help="Path to corpus index database")
    parser.add_argument("-p", "--n-jobs", type=int, default=1, help="Number of workers to run")
    parser.add_argument("-c", "--cross-encoder-name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="SentenceTransformer Cross-Encoder to use for sentence retrieval.")
    parser.add_argument("-i", "--pyserini-index-name", type=str, default="beir-v1.0.0-fever-flat", help="Pyserini index to use for document retrieval. Ignored if skip-document-retrieval flag is added.")
    parser.add_argument("--max-pages", default=5, type=int, help="Number of pages to retain. Ignored if skip-document-retrieval flag is added.")
    parser.add_argument("--max-sentences", default=5, type=int, help="Number of sentences to retain")
    parser.add_argument("--skip-document-retrieval", action="store_true", help="Skip document retrieval, the input data must have 'predicted_pages' field.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exist")
    parser.add_argument("--pipeline-mode", action="store_true", help="Adds 'predicted_evidence' field")
    
    args = parser.parse_args()
    
    infile = Path(args.in_file)
    outfile = Path(args.out_file)
    
    if outfile.exists() and not args.overwrite:
        raise FileExistsError(f"{args.out_file} exist! Use overwrite flag to replace this file.")
    
    doc_retr_path = ""
    if not args.skip_document_retrieval:
        print("Doing document retrieval...")
        doc_retr_path = infile.parent / f"{infile.stem}.n{args.max_pages}.jsonl"
        doc_retr = BM25DocumentRetriever(args.in_file, args.db_path, args.pyserini_index_name, args.max_pages, args.n_jobs)
        doc_retr.batch_document_retrieve()
        doc_retr.to_jsonl(doc_retr_path)
        del doc_retr
        print("End document retrieval...")
    
    print("Do NEI and negative sampling...")
    datain = util.read_data(infile if args.skip_document_retrieval else doc_retr_path)
    db = FeverDocDB(args.db_path)
    ce = CrossEncoder(args.cross_encoder_name)
    
    for doc in tqdm(datain):
        sample = run_sentence_retrieval(doc, db, ce)
        evidence = []
        for i, s in enumerate(sample[1]):
            evidence.append([s])
        if args.pipeline_mode:
            doc["predicted_evidence"] = deepcopy(evidence[:args.max_sentences])
        else:
            if doc["label"] == constants.LOOKUP["label"]["nei"]:
                doc["evidence"] = deepcopy(evidence[:args.max_sentences])
                doc["negative_evidence"] = deepcopy(evidence[args.max_sentences:])
            else:
                negatives = []
                for ev, retr in zip(doc["evidence"], evidence):
                    if ev[0][2] != retr[0][2] or ev[0][3] != retr[0][3]:
                        negatives.append(retr)
                doc["negative_evidence"] = deepcopy(negatives)
            
    print("End NEI and negative sampling...")
    print(f"Writing file to {str(outfile)}")
    util.write_jsonl(outfile, datain)
