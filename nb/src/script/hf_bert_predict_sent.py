import re
from pathlib import Path

from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)

import constants
from gen.util import read_data, write_jsonl


if __name__ == "__main__":
    data_p = Path("/users/k21190024/study/fact-checking-repos/fever/baseline/dumps/bert-data-sent-evidence")
    # data_p = list(data_p.glob("*dev*")) + list(data_p.glob("*test*"))
    # data_p = [p for p in data_p if "fever-climatefever" not in p.stem]
    data_p = [data_p / "scifact.all.test.n5.jsonl"]
    
    model_p = list(Path("/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/models/bert-base-uncased").glob("*sent*.out")) + list(Path("/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/models/xlnet-base-cased").glob("*sent*.out"))
    out_p = Path("/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/predictions/sent")
    # <model-trained-on>-<model-type>.[dev|test].jsonl
    pred_file_fmt = "{0}-{1}.{2}.jsonl"
    
    for model in model_p:
        # output filename components
        model_type = model.parent.stem
        model_trained_on = re.findall("(.*)-{0}.*".format(model_type), model.stem)[0]
        
        # model
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForSequenceClassification.from_pretrained(model)
        pipe = TextClassificationPipeline(model=model_obj, tokenizer=tokenizer, device=0, batch_size=1 if "xlnet" in model_type else 128, function_to_apply="softmax", top_k=3)
        for data in tqdm(data_p, total=len(data_p)):
            split = data.stem.split(".")
            
            dataread = read_data(data)
            data_in = [{"text": doc["evidence"], "text_pair": doc["claim"]} for doc in dataread]
            preds = pipe(data_in, padding=True) if "xlnet" in model_type else pipe(data_in, max_length=512, truncation="only_first", padding=True)
            
            post_preds = []
            for i, d in zip(preds, dataread):
                probas = [0,0,0]
                for j in i:
                    probas[constants.LABEL2ID[j["label"]]] = j["score"]
                post_preds.append({
                    "predicted_label": i[0]["label"], 
                    "predicted_proba": probas, 
                    "claim_id": d["claim_id"]
                })
            
            write_jsonl(out_p / split[0] / pred_file_fmt.format(model_trained_on, model_type, split[1]), post_preds)
            
