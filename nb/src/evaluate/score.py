import json
import six
from pathlib import Path
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

class FEVERScorer:
    """
    Reference: 
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py
    https://github.com/sheffieldnlp/naacl2018-fever/blob/add471f170fbd506bf12f3a778f945f55f3ef8db/src/scripts/score.py
    """
    
    def __init__(self, actual_data, prediction_data, oracle_rte: bool = False, oracle_ir: bool = False, max_evidence: int = 5, score_name: str = ""):
        self._oracle_rte = oracle_rte
        self._oracle_ir = oracle_ir
        self._max_evidence = max_evidence
        self._score_name = score_name
        
        if self._oracle_ir:
            _predicted_sentences = [[[ev[0][2], ev[0][3]] for ev in doc["evidence"]] for doc in actual_data]
            _predicted_pages = [list(set([ev[0][2] for ev in doc["evidence"] if ev[0][2] is not None])) for doc in actual_data]
        else:
            if "predicted_sentences" in prediction_data[0]:
                _predicted_sentences = [doc["predicted_sentences"] for doc in prediction_data]
            else:
                _predicted_sentences = [doc["predicted_evidence"] for doc in prediction_data]

            if "predicted_pages" in prediction_data[0]:
                _predicted_pages = [list(set(doc["predicted_pages"])) for doc in prediction_data]
            else:
                _predicted_pages = []
                for sents in _predicted_sentences:
                    _predicted_pages.append(list(set([p[0] for p in sents])))
        
        if self._oracle_rte:
            _predicted_labels = [doc["label"] for doc in actual_data]
        else:
            if "predicted_label" in prediction_data[0]:
                _predicted_labels = [doc["predicted_label"] for doc in prediction_data]
            else:
                _predicted_labels = [doc["predicted"] for doc in prediction_data]
                
        self.predictions = [{"predicted_evidence": evidence, "predicted_pages": pages, "predicted_label": label} for evidence, pages, label in zip(_predicted_sentences, _predicted_pages, _predicted_labels)]
        
        self.fever_score, self.accuracy, self.precision, self.recall, self.f1 = self.fever_score(self.predictions, actual_data, self._max_evidence)
        self.match_matrix = self.evidence_match_matrix(self.predictions, actual_data)
        
        self.classification_report = None
        self.rte_metrics = None
        if not oracle_rte:
            actual_labels = [doc["label"] for doc in actual_data]
            self.classification_report = classification_report(y_true=actual_labels, y_pred=_predicted_labels)
            mi_p, mi_r, mi_f, _ = precision_recall_fscore_support(y_true=actual_labels, y_pred=_predicted_labels, average="micro", beta=1.0)
            ma_p, ma_r, ma_f, _ = precision_recall_fscore_support(y_true=actual_labels, y_pred=_predicted_labels, average="macro", beta=1.0)
            self.rte_metrics = {
                "fever_score": self.fever_score,
                "accuracy": accuracy_score(y_true=actual_labels, y_pred=_predicted_labels),
                "micro_precision": mi_p,
                "micro_recall": mi_r,
                "micro_f1": mi_f,
                "macro_precision": ma_p,
                "macro_recall": ma_r,
                "macro_f1": ma_f,
                
            }
        
    def __str__(self):
        return f"""
            {self._score_name}
            Oracle IR: {self._oracle_ir}
            Oracle RTE: {self._oracle_rte}
            Max Evidences: {self._max_evidence}

            FEVER Score: {round(self.fever_score * 100, 2)}
            Accuracy: {round(self.accuracy * 100, 2)}
            Macro Precision: {round(self.precision * 100, 2)}
            Macro Recall: {round(self.recall * 100, 2)}
            Macro F1: {round(self.f1 * 100, 2)}
            """
    
    def score_to_dict(self):
        doc_metric = self.get_document_metric()
        key_prefix = "{0}_{1}" if len(self._score_name) > 0 else "{0}{1}"
        return {
            key_prefix.format(self._score_name, "fever_score"): self.fever_score,
            key_prefix.format(self._score_name, "evidence_accuracy"): self.accuracy,
            key_prefix.format(self._score_name, "evidence_precision"): self.precision,
            key_prefix.format(self._score_name, "evidence_recall"): self.recall,
            key_prefix.format(self._score_name, "evidence_f1"): self.f1,
            key_prefix.format(self._score_name, "document_precision"): doc_metric["precision"],
            key_prefix.format(self._score_name, "document_recall"): doc_metric["recall"],
            key_prefix.format(self._score_name, "document_f1"): doc_metric["f1"],
        }

    def check_predicted_evidence_format(self, instance):
        if 'predicted_evidence' in instance.keys() and len(instance['predicted_evidence']):
            assert all(isinstance(prediction, list)
                       for prediction in instance["predicted_evidence"]), \
                "Predicted evidence must be a list of (page,line) lists"

            assert all(len(prediction) == 2
                       for prediction in instance["predicted_evidence"]), \
                "Predicted evidence must be a list of (page,line) lists"

            assert all(isinstance(prediction[0], six.string_types)
                        for prediction in instance["predicted_evidence"]), \
                "Predicted evidence must be a list of (page<string>,line<int>) lists"

            assert all(isinstance(prediction[1], int)
                       for prediction in instance["predicted_evidence"]), \
                "Predicted evidence must be a list of (page<string>,line<int>) lists"


    def is_correct_label(self, instance):
        return instance["label"].upper() == instance["predicted_label"].upper()


    def is_strictly_correct(self, instance, max_evidence=None):
        #Strict evidence matching is only for NEI class
        self.check_predicted_evidence_format(instance)

        if instance["label"].upper() != "NOT ENOUGH INFO" and self.is_correct_label(instance):
            assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

            if max_evidence is None:
                max_evidence = len(instance["predicted_evidence"])


            for evience_group in instance["evidence"]:
                #Filter out the annotation ids. We just want the evidence page and line number
                actual_sentences = [[e[2], e[3]] for e in evience_group]
                #Only return true if an entire group of actual sentences is in the predicted sentences
                if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                    return True

        #If the class is NEI, we don't score the evidence retrieval component
        elif instance["label"].upper() == "NOT ENOUGH INFO" and self.is_correct_label(instance):
            return True

        return False


    def evidence_macro_precision(self, instance, max_evidence=None):
        this_precision = 0.0
        this_precision_hits = 0.0

        if instance["label"].upper() != "NOT ENOUGH INFO":
            all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

            predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                            instance["predicted_evidence"][:max_evidence]

            for prediction in predicted_evidence:
                if prediction in all_evi:
                    this_precision += 1.0
                this_precision_hits += 1.0

            return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

        return 0.0, 0.0

    def evidence_macro_recall(self, instance, max_evidence=None):
        # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
        if instance["label"].upper() != "NOT ENOUGH INFO":
            # If there's no evidence to predict, return 1
            if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
                return 1.0, 1.0

            predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                            instance["predicted_evidence"][:max_evidence]

            for evidence_group in instance["evidence"]:
                evidence = [[e[2], e[3]] for e in evidence_group]
                if all([item in predicted_evidence for item in evidence]):
                    # We only want to score complete groups of evidence. Incomplete groups are worthless.
                    return 1.0, 1.0
            return 0.0, 1.0
        return 0.0, 0.0


    # Micro is not used. This code is just included to demostrate our model of macro/micro
    def evidence_micro_precision(self, instance):
        this_precision = 0
        this_precision_hits = 0

        # We only want to score Macro F1/Precision/Recall of recalled evidence for NEI claims
        if instance["label"].upper() != "NOT ENOUGH INFO":
            all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

            for prediction in instance["predicted_evidence"]:
                if prediction in all_evi:
                    this_precision += 1.0
                this_precision_hits += 1.0

        return this_precision, this_precision_hits


    def fever_score(self, predictions,actual=None, max_evidence=5):
        correct = 0
        strict = 0

        macro_precision = 0
        macro_precision_hits = 0

        macro_recall = 0
        macro_recall_hits = 0

        for idx,instance in enumerate(predictions):
            assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

            #If it's a blind test set, we need to copy in the values from the actual data
            if 'evidence' not in instance or 'label' not in instance:
                assert actual is not None, 'in blind evaluation mode, actual data must be provided'
                assert len(actual) == len(predictions), 'actual data and predicted data length must match'
                assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
                instance['evidence'] = actual[idx]['evidence']
                instance['label'] = actual[idx]['label']

            assert 'evidence' in instance.keys(), 'gold evidence must be provided'

            if self.is_correct_label(instance):
                correct += 1.0

                if self.is_strictly_correct(instance, max_evidence):
                    strict+=1.0

            macro_prec = self.evidence_macro_precision(instance, max_evidence)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = self.evidence_macro_recall(instance, max_evidence)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        total = len(predictions)

        strict_score = strict / total
        acc_score = correct / total

        pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

        f1 = 2.0 * pr * rec / (pr + rec)

        return strict_score, acc_score, pr, rec, f1
    
    def count_evidence_sent_miss(self, instance, evidence_type: str = "evidence"):
        if instance[evidence_type] is not None:
            all_evidences = defaultdict(list)
            for eg in instance[evidence_type]:
                for e in eg:
                     if e[3] is not None:
                        all_evidences[e[2]].append(e[3])

            ev_sent_miss = 0
            for p_eg in instance["predicted_evidence"]:
                sents = all_evidences.get(p_eg[0], [])
                if sents and p_eg[1] not in sents:
                    ev_sent_miss += 1
            return ev_sent_miss
        else:
            return None
        
    def count_evidence_full_hit(self, instance, evidence_type: str = "evidence"):
        if instance[evidence_type] is not None:
            full_hits = 0
            all_evi = [[e[2], e[3]] for eg in instance[evidence_type] for e in eg if e[3] is not None]

            for prediction in instance["predicted_evidence"]:
                if prediction in all_evi:
                    full_hits += 1.0

            return full_hits
        else:
            return None
    
    def evidence_match_matrix(self, predictions, actual):
        """
        unique page hit, page hit and sentence hit
        
        full_hit = page hit and sentence hit in "evidence"
        evidence_sent_miss = page hit and sentence miss in "evidence" 
        """
        
        match_matrix = []
        for idx, instance in enumerate(predictions):
            instance["evidence"] = actual[idx]["evidence"]
            instance["label"] = actual[idx]["label"]
            
            match_d = {
                "claim_id": actual[idx]["id"],
                "claim_label": actual[idx]["label"],
                "n_predicted_evidences": len(instance["predicted_evidence"]),
                "n_total_evidences": len([e[2] for eg in instance["evidence"] for e in eg if e[3] is not None])
            }
            
            # full hit
            match_d["full_hit"] = self.count_evidence_full_hit(instance, "evidence")
            # page hit and sentence miss
            match_d["evidence_sent_miss"] = self.count_evidence_sent_miss(instance, "evidence")
            # irrelevant pages
            unique_evidence_pages = set([e[2] for eg in instance["evidence"] for e in eg if e[3] is not None])
            predicted_unique_evidence_pages = set(instance["predicted_pages"])
            match_d["irrelevant"] = predicted_unique_evidence_pages.difference(unique_evidence_pages)
            
            match_d["evidence_page"] = unique_evidence_pages
            match_d["predicted_page"] = predicted_unique_evidence_pages

            match_matrix.append(match_d)
        
        return match_matrix
    
    def get_document_metric(self, match_matrix = None, return_df: bool = False):
        """
        Get document accuracy, macro recall and macro precision
        """
        df_mm = pd.DataFrame(self.match_matrix if match_matrix is None else match_metrix)
        df_mm["_tp"] = df_mm[["evidence_page", "predicted_page"]].apply(lambda x: x[0] & x[1], axis=1).apply(len)
        # df_mm["doc_accuracy"] = df_mm["_tp"] / df_mm[["evidence_page", "predicted_page"]].apply(lambda x: x[0] | x[1], axis=1).apply(len)
        df_mm["doc_recall"] = df_mm["_tp"] / df_mm["evidence_page"].apply(len)
        df_mm["doc_precision"] = df_mm["_tp"] / df_mm["predicted_page"].apply(len)
        
        metrics = {
            # "accuracy": df_mm.loc[df_mm["claim_label"] != "NOT ENOUGH INFO", "doc_accuracy"].mean(), 
            "recall": df_mm.loc[df_mm["claim_label"] != "NOT ENOUGH INFO", "doc_recall"].mean(), 
            "precision": df_mm.loc[df_mm["claim_label"] != "NOT ENOUGH INFO", "doc_precision"].mean()
        }
        metrics["f1"] = 2.0 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        
        return (metrics, df_mm) if return_df else metrics


class ClimateFEVERScorer(FEVERScorer):
    """
    Modify FEVER scorer such that NEI claims are also evaluated since Climate-FEVER provides
    actual NEI evidences
    """
    def __init__(self, actual_data, prediction_data, oracle_rte: bool = False, oracle_ir: bool = False, max_evidence: int = 5, score_name: str = ""):
        super().__init__(actual_data, prediction_data, oracle_rte, oracle_ir, max_evidence, score_name)
        
    def is_strictly_correct(self, instance, max_evidence=None):
        #Strict evidence matching is only for NEI class
        self.check_predicted_evidence_format(instance)

        if self.is_correct_label(instance):
            assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

            if max_evidence is None:
                max_evidence = len(instance["predicted_evidence"])

            for evidence_group in instance["evidence"]:
                #Filter out the annotation ids. We just want the evidence page and line number
                actual_sentences = [[e[2], e[3]] for e in evidence_group]
                #Only return true if an entire group of actual sentences is in the predicted sentences
                if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                    return True

        return False


    def evidence_macro_precision(self, instance, max_evidence=None):
        this_precision = 0.0
        this_precision_hits = 0.0

        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    def evidence_macro_recall(self, instance, max_evidence=None):
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0

    def fever_score(self, predictions, actual=None, max_evidence=5):
        correct = 0
        strict = 0

        macro_precision = 0
        macro_precision_hits = 0

        macro_recall = 0
        macro_recall_hits = 0

        for idx,instance in enumerate(predictions):
            assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

            #If it's a blind test set, we need to copy in the values from the actual data
            if 'evidence' not in instance or 'label' not in instance:
                assert actual is not None, 'in blind evaluation mode, actual data must be provided'
                assert len(actual) == len(predictions), 'actual data and predicted data length must match'
                assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
                instance['evidence'] = actual[idx]['evidence']
                instance['label'] = actual[idx]['label']

            assert 'evidence' in instance.keys(), 'gold evidence must be provided'

            if self.is_correct_label(instance):
                correct += 1.0

                if self.is_strictly_correct(instance, max_evidence):
                    strict+=1.0

            macro_prec = self.evidence_macro_precision(instance, max_evidence)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = self.evidence_macro_recall(instance, max_evidence)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        total = len(predictions)

        strict_score = strict / total
        acc_score = correct / total

        pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

        f1 = 2.0 * pr * rec / (pr + rec)

        return strict_score, acc_score, pr, rec, f1
    
    def evidence_match_matrix(self, predictions, actual):
        """
        unique page hit, page hit and sentence hit
        
        full_hit = page hit and sentence hit in "evidence"
        evidence_sent_miss = page hit and sentence miss in "evidence" 
        other_evidence_full_hit = page hit and sentence hit in "other_evidence"
        other_evidence_sent_miss = page hit and sentence miss in "other_evidence"
        irrelevant = page predicted but neither "evidence" or "other_evidence" has the page
        """
        
        match_matrix = []
        for idx, instance in enumerate(predictions):
            instance["evidence"] = actual[idx]["evidence"]
            instance["other_evidence"] = actual[idx]["other_evidence"]
            instance["label"] = actual[idx]["label"]
            
            match_d = {
                "claim_id": actual[idx]["id"],
                "claim": actual[idx]["claim"],
                "claim_label": actual[idx]["label"],
                "n_predicted_evidences": len(instance["predicted_evidence"]),
                "n_total_evidences": len([e[2] for eg in instance["evidence"] for e in eg if e[3] is not None])
            }
            
            # full hit
            match_d["full_hit"] = self.count_evidence_full_hit(instance, "evidence")
            match_d["other_evidence_full_hit"] = self.count_evidence_full_hit(instance, "other_evidence")
            
            # page hit and sentence miss
            match_d["evidence_sent_miss"] = self.count_evidence_sent_miss(instance, "evidence")
            match_d["other_evidence_sent_miss"] = self.count_evidence_sent_miss(instance, "other_evidence")
            
            # irrelevant pages
            unique_evidence_pages = set([e[2] for eg in instance["evidence"] for e in eg if e[3] is not None])
            unique_other_evidence_pages = set() if instance["other_evidence"] is None else set([e[2] for eg in instance["other_evidence"] for e in eg if e[3] is not None])
            predicted_unique_evidence_pages = set(instance["predicted_pages"])
            match_d["irrelevant"] = predicted_unique_evidence_pages.difference(unique_evidence_pages.union(unique_other_evidence_pages))
            
            match_d["evidence_page"] = unique_evidence_pages
            match_d["predicted_page"] = predicted_unique_evidence_pages
            
            match_matrix.append(match_d)
        
        return match_matrix
    
    def get_document_metric(self, match_matrix = None, return_df: bool = False):
        """
        Get document accuracy, macro recall and macro precision
        """
        df_mm = pd.DataFrame(self.match_matrix if match_matrix is None else match_metrix)
        df_mm["_tp"] = df_mm[["evidence_page", "predicted_page"]].apply(lambda x: x[0] & x[1], axis=1).apply(len)
        # df_mm["doc_accuracy"] = df_mm["_tp"] / df_mm[["evidence_page", "predicted_page"]].apply(lambda x: x[0] | x[1], axis=1).apply(len)
        df_mm["doc_recall"] = df_mm["_tp"] / df_mm["evidence_page"].apply(len)
        df_mm["doc_precision"] = df_mm["_tp"] / df_mm["predicted_page"].apply(len)
        
        metrics = {
            # "accuracy": df_mm["doc_accuracy"].mean(), 
            "recall": df_mm["doc_recall"].mean(), 
            "precision": df_mm["doc_precision"].mean()
        }
        metrics["f1"] = 2.0 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        
        return (metrics, df_mm) if return_df else metrics
