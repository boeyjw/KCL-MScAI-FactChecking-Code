import json
from pathlib import Path
from collections import defaultdict
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_finetune(model_path: Path, metric: List[str] = ["loss", "accuracy", "macro_fscore"], return_metric: bool = False):
    """
    Plot fine tuning results generated by the output of allennlp==0.9.0
    """
    metrics = []

    for p in model_path.glob("metrics_epoch_*.json"):
        with p.open("r") as fn:
            m = json.load(fn)
            metrics.append(m)
    metrics = pd.DataFrame(metrics)
    metrics = metrics.sort_values("epoch")
    
    fig, ax = plt.subplots(1, len(metric), figsize=[10, 3])
    for i, m in enumerate(metric):
        for c in metrics.filter(like=m).columns:
            if "best" not in c:            
                ax[i] = metrics.plot(x="epoch", y=c, ax=ax[i])
        _ = ax[i].set_ylabel(m)

    return (ax, metrics) if return_metric else ax

def plot_model_confidence(actual, predicted):
    """
    Plot model confidence only for models that has a softmax output layer
    """
    assert len(actual) == len(predicted)
    
    correct_preds, incorrect_preds = defaultdict(list), defaultdict(list)
    for yt, yp in zip(actual, predicted):
        if yt["label"] == yp["predicted_label"]:
            # model confidence on making the right decision
            correct_preds[yt["label"]].append(max(yp["label_probs"]))
        else:
            # model confidence on making the wrong decision
            incorrect_preds[yp["predicted_label"]].append(max(yp["label_probs"]))
            
    fig, ax = plt.subplots(1, 3, figsize=[10, 4])
    fig.suptitle("Correct Prediction Model Confidence")
    for i, label in enumerate(["SUPPORTS", "NOT ENOUGH INFO", "REFUTES"]):
        _ = sns.kdeplot(x=correct_preds[label], bw_adjust=.3, cut=1, ax=ax[i])
        ax[i].set_title(label)
        if i > 0:
            ax[i].set_ylabel("")
            
    fig, ax = plt.subplots(1, 3, figsize=[10, 4])
    fig.suptitle("Incorrect Prediction Model Confidence")
    for i, label in enumerate(["SUPPORTS", "NOT ENOUGH INFO", "REFUTES"]):
        _ = sns.kdeplot(x=incorrect_preds[label], bw_adjust=.3, cut=1, ax=ax[i])
        ax[i].set_title(label)
        if i > 0:
            ax[i].set_ylabel("")
    return correct_preds, incorrect_preds

def plot_evidence_curves(df, x_is_page: bool, metric: str, suptitle: str, bold_line_index: List[int] = [-1]):
    """
    Plot metric of IR component
    """
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])

    fig.suptitle(suptitle)

    for idx, split in enumerate(["train", "dev"]):
        ax[idx].set_title(split)
        ax[idx].set_xlabel("Pages" if x_is_page else "Sentences")
        ax[idx].set_ylabel(metric)
        for i in range(1, 11):
            df_plt = df.loc[df["n_sents" if x_is_page else "n_pages"] == i, ["n_pages" if x_is_page else "n_sents", f"{split}_{metric}"]]
            if idx == 0:
                ax[idx].plot(
                    df_plt["n_pages" if x_is_page else "n_sents"], 
                    df_plt[f"{split}_{metric}"], 
                    label="{0} {1}".format(i, "Sentence" if x_is_page else "Pages"),
                    linewidth=3 if i in bold_line_index else 1
                )
            else:
                ax[idx].plot(
                    df_plt["n_pages" if x_is_page else "n_sents"], 
                    df_plt[f"{split}_{metric}"],
                    linewidth=3 if i in bold_line_index else 1
                )

    fig.legend()
    
    return fig, ax