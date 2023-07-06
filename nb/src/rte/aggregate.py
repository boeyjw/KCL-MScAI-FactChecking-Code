from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import softmax


ID2LABEL = {0: "SUPPORTS", 1: "NOT ENOUGH INFO", 2: "REFUTES"}

def agg_predict_proba(proba):
    """
    Mean softmax probability across labels. Returns the label
    with the highest probability. If labels have equal probability,
    return the first one in the order of SUPPORTS, NOT ENOUGH INFO, REFUTES
    """
    probas = np.array([pr for pr in proba])
    return np.argmax(probas.mean(axis=0))

def agg_predict(preds):
    """
    Climate-FEVER style majority aggregation
    SUPPORTS if evidence only contain SUPPORTS and NOT ENOUGH INFO
    REFUTES  if evidence only contain REFUTES and NOT ENOUGH INFO
    NOT ENOUGH INFO if evidence only contain NOT ENOUGH INFO
    
    If evidence contain both SUPPORT and REFUTES, then the majority
    label is taken as final label.
    If evidence contain both SUPPORT and REFUTES in equal numbers,
    then return NOT ENOUGH INFO
    """
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
    # FIXME: For climate-fever, this aggregation is wrong for actual data
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