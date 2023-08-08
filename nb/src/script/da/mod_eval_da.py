# Reference: https://github.com/sheffieldnlp/naacl2018-fever/blob/master/src/scripts/rte/da/eval_da.py

import os

from copy import deepcopy
from pathlib import Path
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#from allennlp.common.util import prepare_environment
from allennlp.data import Tokenizer
from allennlp.models import load_archive
from rte.da.util_log_helper import LogHelper
from rte.da.mod_reader import FEVERReader
from tqdm import tqdm
import argparse
import logging
import json
import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def eval_model(args):
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()

    # COMMENT: token_indexers = None uses the default SingleID Tokenizer
    reader = FEVERReader(wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                        claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                        token_indexers=None)

    logger.info("Reading training data from %s", args.in_file)
    data = reader.read(args.in_file).instances
    raw_data = reader.raw_read_data(Path(args.in_file))

    if args.log is not None:
        f = open(args.log,"w+")

    for item, raw in tqdm(zip(data, raw_data)):
        prediction = model.forward_on_instance(item)
        cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]

        if args.log is not None:
            res = {
                "predicted_logits":prediction["label_logits"].tolist(), 
                "predicted":cls, 
                "predicted_proba": prediction["label_probs"].tolist(),
                "claim_id": raw["claim_id"]
            }
            if "predicted_evidence" in raw and raw["predicted_evidence"] is not None:
                res["predicted_evidence"] = raw["predicted_evidence"]
            f.write(json.dumps(res)+"\n")

    if args.log is not None:
        f.close()

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('archive_file', type=str, help='Path to AllenNLP DA model')
    parser.add_argument('in_file', type=str, help='Path to data to predict')
    parser.add_argument('log', type=str, help='Path to write predictions.')

    parser.add_argument("--cuda-device", type=int, default=0, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')



    args = parser.parse_args()
    eval_model(args)
