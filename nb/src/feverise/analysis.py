from collections import defaultdict

import numpy as np
import scipy

def count_evidences(claims):
    def _count_statistics(evidence_count):
        cnt = np.array(evidence_count)
        return np.min(cnt), scipy.stats.mode(cnt, keepdims=False), np.mean(cnt), np.max(cnt)
    
    n_evidences = defaultdict(list)
    for d in claims:
        n_evidences[d["label"]].append(0)
        for i in d["evidence"]:
            if i[0][2] is not None:
                n_evidences[d["label"]][-1] += 1
        if "other_evidence" in d and d["other_evidence"] is not None:
            for i in d["other_evidence"]:
                if i[0][2] is not None:
                    n_evidences[d["label"]][-1] += 1
    
    stats = {k: _count_statistics(cnt) for k, cnt in n_evidences.items()}
    
    return n_evidences, stats