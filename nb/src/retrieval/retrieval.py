from pathlib import Path
from copy import deepcopy

from pyserini.search.lucene import LuceneSearcher

from gen import util
from .fever_doc_db import FeverDocDB

class Retriever(object):
    def __init__(self, data_path, db_path):
        if isinstance(db_path, Path):
            db_path = str(db_path.resolve())
        self.db = FeverDocDB(db_path)
        self.data = util.read_data(Path(data_path))
        self.results = None

    def _check_docid_exist(self, docid):
        lines = self.db.get_doc_lines(util.denormalise_title(docid))
        return lines is not None
    
    def to_jsonl(self, out_path, overwrite: bool = False):
        out_path = Path(out_path)
        if out_path.exist() and not overwrite:
            raise FileExistsError("Result file {0} exists! Switch overwrite to True to replace file.".format(out_path))
        util.write_jsonl(out_path, self.results)
    

class BM25DocumentRetriever(Retriever):
    def __init__(
        self, data_path, db_path, pyserini_index_name, 
        bm25_top_k: int = 100, n_jobs: int = 1
    ):
        super().__init__(data_path, db_path)
        self.results = []
        self.bm25_searcher = LuceneSearcher.from_prebuilt_index(pyserini_index_name)
        self.bm25_searcher.set_bm25()
        
        self.bm25_top_k = bm25_top_k
        self.n_jobs = n_jobs
        
    def batch_document_retrieve(self):
        q, qids = [], []
        for idx, doc in enumerate(self.data):
            q.append(doc["claim"])
            qids.append(str(doc["id"]))
        
        # retrieve documents
        retrieved_docs = self.bm25_searcher.batch_search(q, qids=qids, k=self.bm25_top_k, threads=self.n_jobs)
        # check if document is available in database
        retrieved_docs = {
            k: [[i.docid, float(i.score), self._check_docid_exist(i.docid)] for i in v] 
            for k, v in retrieved_docs.items()
        }
        # update the data
        for doc in self.data:
            retr_doc = deepcopy(doc)
            retr_doc["predicted_pages_score"] = retrieved_docs[str(doc["id"])]
            # only add pages that exist in the db as predicted pages
            retr_doc["predicted_pages"] = [i[0] for i in retrieved_docs[str(doc["id"])] if i[2]]
            self.results.append(retr_doc)
        