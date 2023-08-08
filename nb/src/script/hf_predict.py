import re
import argparse
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)

import constants
from gen.util import read_data, write_jsonl
from rte import aggregate

def aggregate_preds(preds):
    df_grp = pd.DataFrame(preds)
    has_pred_ev = "predicted_evidence" in df_grp
    
    df_grp["predicted_label"] = df_grp["predicted_label"].map(constants.LABEL2ID)
    df_grp = df_grp.groupby("claim_id", sort=False, as_index=False)
    
    df_majority = df_grp.agg({"predicted_label": aggregate.agg_predict})
    df_majority["predicted_label"] = df_majority["predicted_label"].map(constants.ID2LABEL)
    
    df_mean = df_grp.agg({"predicted_proba": aggregate.agg_predict_proba}).rename(columns={"predicted_proba": "predicted_label"})
    df_mean["predicted_label"] = df_mean["predicted_label"].map(constants.ID2LABEL)
    
    if has_pred_ev:
        df_ev = df_grp.agg({"predicted_evidence": list})
        df_majority = df_majority.merge(df_ev, on="claim_id", how="left")
        df_mean = df_mean.merge(df_ev, on="claim_id", how="left")
    
    return df_majority.to_dict("records"), df_mean.to_dict("records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BERT and XLNet Batch Prediction Script")
    parser.add_argument("data_dir", type=str, help="Data to infer directory")
    parser.add_argument("model_dir", type=str, help="Model directory")
    parser.add_argument("out_dir", type=str, help="Directory to write predictions")
    parser.add_argument("--cuda-device", type=int, default=-1, help="CUDA device ID. -1 for CPU, positive integer to indicate GPU ID")
    parser.add_argument("--document", action="store_true", help="Document modelling predictions. Leave this out for sentence modelling")
    parser.add_argument("--aggregate", action="store_true", help="Do majority and mean probability aggregation. Ignored when used with --document")
    
    args = parser.parse_args()
    if args.document:
        warnings.warn("Using --document. Ignoring --aggregate flag...")
        args.aggregate = False
    
    data_p = Path(args.data_dir)
    data_p = [p for p in data_p.iterdir() if p.is_file() and "fever-climatefever" not in p.stem]
    if len(data_p) == 0:
        raise FileNotFoundError("Data directory is empty!")
    
    model_p = list(Path(args.model_dir).iterdir())
    if len(model_p) == 0:
        raise FileNotFoundError("Model directory is empty!")
        
    out_p = Path(args.out_dir)
    out_p.mkdir(exist_ok=True)
    
    # <model-trained-on>-<model-type>.[dev|test].jsonl
    pred_file_fmt = "{0}-{1}.{2}.jsonl"
    
    for model in model_p:
        # output filename components
        model_type = "xlnet-base-cased" if "xlnet" in model.stem else "bert-base-uncased"
        model_trained_on = re.findall("(.*)-{0}.*".format(model_type), model.stem)[0]
        
        # model
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForSequenceClassification.from_pretrained(model)
        pipe = TextClassificationPipeline(
            model=model_obj, 
            tokenizer=tokenizer, 
            device=args.cuda_device, 
            batch_size=1 if "xlnet" in model_type else 32, 
            function_to_apply="softmax", 
            top_k=3
        )
        for data in tqdm(data_p, total=len(data_p)):
            split = data.stem.split(".")
            
            dataread = read_data(data)
            data_in = [{"text": doc["evidence"], "text_pair": doc["claim"]} for doc in read_data(data)]
            
            if args.document:
                preds = pipe(
                    data_in, 
                    max_length=1024 if "xlnet" in model_type else 512, 
                    truncation="only_first", 
                    padding=True
                )
            else:
                if "xlnet" in model_type:
                    preds = pipe(data_in, padding=True)  
                else :
                    preds = pipe(data_in, max_length=512, truncation="only_first", padding=True)
            
            post_preds = []
            for i, d in zip(preds, dataread):
                probas = [0,0,0]
                for j in i:
                    probas[constants.LABEL2ID[j["label"]]] = j["score"]
                res = {
                    "predicted_label": i[0]["label"], 
                    "predicted_proba": probas,
                    "claim_id": d["claim_id"]
                }
                if "predicted_evidence" in d and d["predicted_evidence"] is not None:
                    res["predicted_evidence"] = d["predicted_evidence"]
                post_preds.append(res)
            
            write_p = out_p / split[0]
            write_p.mkdir(exist_ok=True)
            if args.document or not args.aggregate:
                write_p = write_p / pred_file_fmt.format(model_trained_on, model_type, split[1])
                write_jsonl(write_p, post_preds)
            elif args.aggregate:
                maj, mp = aggregate_preds(post_preds)
                
                maj_p = write_p / "majority" / pred_file_fmt.format(model_trained_on, model_type, split[1])
                maj_p.parent.mkdir(exist_ok=True)
                write_jsonl(maj_p, maj)

                mp_p = write_p / "meanproba" / pred_file_fmt.format(model_trained_on, model_type, split[1])
                mp_p.parent.mkdir(exist_ok=True)            
                write_jsonl(mp_p, mp)
            
