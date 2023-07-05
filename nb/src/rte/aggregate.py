from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import softmax


ID2LABEL = {0: "SUPPORTS", 1: "NOT ENOUGH INFO", 2: "REFUTES"}

def agg_predict_proba(probas):
    return np.argmax(np.array(probas).mean())

def agg_predict(preds):
    preds = preds.values
    if 0 in preds and 2 in preds:
        cnt = Counter(preds)
        if cnt[0] > cnt[2]:
            return 0
        elif cnt[0] < cnt[2]:
            return 2
        else:
            return 1
    elif 0 in preds:
        return 0
    elif 2 in preds:
        return 2
    else:
        return 1

def generate_micro_macro_df(actual, preds):
    df_micro_preds = pd.DataFrame({
        "claim": actual["claim"],
        "evidence": actual["evidence"],
        "actual": actual["labels"],
        "predicted": [np.argmax(logit) for logit in preds.predictions],
        "proba": [proba for proba in softmax(preds.predictions, axis=1)]
    })
    df_macro_preds = (
        df_micro_preds
        .groupby("claim")
        .agg({"actual": "max", "proba": agg_predict_proba, "predicted": agg_predict})
    )
    
    df_micro_preds["actual"] = df_micro_preds["actual"].map(ID2LABEL)
    df_micro_preds["predicted"] = df_micro_preds["predicted"].map(ID2LABEL)
    df_macro_preds["actual"] = df_macro_preds["actual"].map(ID2LABEL)
    df_macro_preds["proba"] = df_macro_preds["proba"].map(ID2LABEL)
    df_macro_preds["predicted"] = df_macro_preds["predicted"].map(ID2LABEL)
    
    return df_micro_preds, df_macro_preds

def generate_doc_df(actual, preds):
    df = pd.DataFrame({
        "claim": actual["claim"],
        "evidence": actual["evidence"],
        "actual": actual["labels"],
        "predicted": [np.argmax(logit) for logit in preds.predictions]
    })
    df["actual"] = df["actual"].map(ID2LABEL)
    df["predicted"] = df["predicted"].map(ID2LABEL)
    
    return df