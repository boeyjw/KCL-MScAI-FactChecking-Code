# Robustness of Fact-Checking Models: Train on FEVER and Climate-FEVER, Evaluate on SciFact

I verify that I am the sole author of the programmes contained in this archive, except where explicitly stated to the contrary

Boey Jian Wen (11 August 2023)

# General Structure

```bash
ROOT/
##########################  TeX Source Report
├── 7CCSMPRJ-Report/
##########################  Processing output
├── data/
├── evidence_corpus_dbs/
##########################  Prediction output
├── predictions/
├── metrics/
├── submissions/
##########################  Trained Models
├── models
│   └──da-document-models/
│   └──document-models/
│   └──sentence-models/
##########################  Word embedding used for DA model
├── glove/
##########################  Source code
├── maincode/
├── src/
##########################  Anaconda/Pip Environment
├── daenv.txt
├── mainenv.yml
```

In Supplementary zip file: 7CCSMPRJ-Report, data, predictions, metrics, submissions, maincode, src, daenv.txt and mainenv.yml

In [OneDrive](https://emckclac-my.sharepoint.com/:f:/g/personal/k21190024_kcl_ac_uk/EmqI941nuXlKhvduax46fs0BWs0w5QaO2EwlVo0L-A0MKg?e=IbmAHd): da-document-models, document-models, sentence-models, glove

Report repository: https://github.kcl.ac.uk/k21190024/7CCSMPRJ-Report

Code repository: https://github.kcl.ac.uk/k21190024/7CCSMPRJ-Code

All the processed data used alongside model training and evaluation were provided. Nonetheless, the experiments can be replicated by following the guide below.

# Experimental Steps
This guide assumes all the General structure folders are located in a single directory. Most of the scripts also makes the same assumption and may require path changes to work if the assumption does not hold. All code were executed in Ubuntu 20.04 LTS and some packages/script may not be compatible with non-linux OS.

## 1. Download raw data
| Dataset | Link |
|---------|------|
| FEVER | https://fever.ai/dataset/fever.html |
| Climate-FEVER | https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html |
| SciFact | https://github.com/allenai/scifact |

Unpack all the datasets and store them under ```data/```.

## 2. Create environment
The main environment can be recreated using Anaconda with the environment file using command:
```bash
conda env create -f mainenv.yml
``` 

PyTorch and faiss may fail to install and halts the environment creation due to GPU differences. It is recommended to install the two packages individually then rerun the command.

The AllenNLP environment must be recreated manually by first creating a new environment running Python 3.6:
```bash
conda create -n daenv python=3.6 pip
```

Then, each line in ```daenv.txt``` __must be installed using pip in order__. PyTorch can be installed using conda but it must be PyTorch 1.9.1. After creating the environment, copy adapted DA model file to AllenNLP library using command ```cp src/script/da/decomposable_attention.py /path/to/environment/lib/python3.6/site-packages/allennlp/models/```. This updates the model to track F1 macro and micro as well.

## 3. Run code
```bash
ROOT/
├── maincode/
│   ├── 00_Feverise.ipynb
│   ├── 01_Split_Climate-FEVER.ipynb
│   ├── 03_Model_Training/
│   └── 04_Model_Evaluation/
├── src/
│   ├── __init__.py
│   ├── constants.py  # Constant variables
│   ├── feverise/  # Preprocessing data to FEVER format
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   ├── build_db.py  # Build DrQA DB
│   │   ├── build_db_mod.py  # Build DrQA DB but with title_id instead
│   │   ├── climatefever.py  # Climate-FEVER Pure converter
│   │   ├── climatefever_sent.py  # Climate-FEVER converter
│   │   ├── scifact.py  # SciFact converter
│   │   └── util.py  # id to title_id conversion
│   ├── gen/  # General utility functions
│   │   ├── __init__.py
│   │   ├── special.py  # Entropy calculation
│   │   └── util.py  # General utilities
│   ├── retrieval/  # Evidence retrieval code
│   │   ├── __init__.py
│   │   ├── doc_db.py  # Borrowed DrQA document DB code
│   │   ├── fever_doc_db.py  # Adapted doc_db code
│   │   └── retrieval.py  # BM25 document and sentence retrieval
│   ├── rte/  # RTE code
│   │   ├── aggregate.py  # Majority and mean probability aggregation
│   │   └── da/  # Borrowed DA training and inference code
│   │       ├── common_options.py
│   │       ├── mod_reader.py
│   │       ├── util_log_helper.py
│   │       └── util_random.py
│   ├── scoring/  # Evaluation object code
│   │   ├── __init__.py
│   │   ├── chart.py  # Plot DA fine-tuning charts
│   │   └── scorer.py  # Adapted FEVER scoring object
│   ├── script/  # Processing sripts
│   │   ├── da/  # Specific DA scripts
│   │   │   ├── batch_da_predict.py  # Batch DA inference
│   │   │   ├── decomposable_attention.py  # Adapted AllenNLP code to include F1 metrics
│   │   │   ├── mod_eval_da.py  # Adapted DA evaluation script
│   │   │   ├── mod_train_da.py  # Adapted DA training script
│   │   │   ├── parameters/  # Adapted DA hyperparameter files
│   │   ├── __init__.py
│   │   ├── batch_prepare_docsent.sh  # Batch NEI sampling, document and sentence preperation
│   │   ├── evidence_sampling.py  # NEI sampling
│   │   ├── hf_predict.py  # BERT and XLNet inference
│   │   ├── prepare_docsent_data.py  # Prepare document and sentence data
│   │   └── scifact_submission.py  # Convert predictions to SciFact format
│   └── submission/
│       └── scifact.py  # Postprocess predictions to SciFact format
```
### Preprocessing
1. Build FEVER evidence corpus using command:
```bash
PYTHONPATH=src python src/feverise/build_db.py data/fever/wiki-pages evidence_corpus_dbs/fever.db
```
2. Run ```maincode/00_Feverise.ipynb```: Preprocess Climate-FEVER, Climate-FEVER Pure and SciFact into FEVER format and build their evidence corpus DB.

3. Run ```maincode/01_Split_Climate-FEVER.ipynb```: Split Climate-FEVER and Climate-FEVER Pure dataset into train/dev/test.

4. Run ```bash src/script/batch_prepare_docsent.sh```: Perform NEI sampling on FEVER and SciFact and generate document and sentence model datasets. Data folder structure should now be:
```bash
ROOT/
├── data/
│   ├── doc-dataset/
│   ├── fever-nei-sampled/
│   ├── feverised-climatefever/
│   ├── feverised-climatefever_sent/
│   ├── feverised-scifact/
│   ├── scifact-nei-sampled/
│   ├── scifact-pipeline-sampled/
│   ├── scifact-test-sampled/
│   └── sent-dataset/
├── evidence_corpus_dbs/
│   ├── fever.db
│   ├── feverised-climatefever-titleid.db
│   ├── feverised-scifact.db
```

Manual run can also be done by using the below commands as reference:

NEI sampling
```bash
PYTHONPATH=src python src/script/evidence_sampling.py /path/to/dataset /path/to/write /path/to/evidence_corpus.db -p num_process --max-pages max_retrieved_evidence_document --max-sentences max_retrieved_evidence -i pyserini-index-name -c sentence-transformers-cross_encoder_name 
```
Append ```--pipeline-mode``` if "predicted_evidence" field is required

Prepare document data
```bash
PYTHONPATH=src python src/script/prepare_docsent_data.py /path/to/nei-sampled-dataset /path/to/write /path/to/evidence_corpus.db -p num_process --id-prefix prefix|id -c sentence-transformers-cross_encoder_name --max-evidence filter_to_max_evidence --nei-max-evidence filter_nei_claims_to_max_evidence
```
Append ```--pipeline-mode``` if "predicted_evidence" field is required. ```--max-evidence --nei-max-evidence``` can be ignored if all evidences are to be kept. Without ```--id-prefix```, the first token split by "-" folder name which the data reside will be used as the prefix.

Prepare sentence data
```bash
PYTHONPATH=src python src/script/prepare_docsent_data.py /path/to/nei-sampled-dataset /path/to/write /path/to/evidence_corpus.db --sentence-mode -p num_process --id-prefix prefix|id -c sentence-transformers-cross_encoder_name  --max-evidence filter_to_max_evidence --nei-max-evidence filter_nei_claims_to_max_evidence 
```
Append ```--pipeline-mode``` if "predicted_evidence" field is required. ```--max-evidence --nei-max-evidence``` can be ignored if all evidences are to be kept. Without ```--id-prefix```, the first token split by "-" folder name which the data reside will be used as the prefix.

Prepare FEVER + Climate-FEVER \[Pure\] data
```bash
cat fever.[split].jsonl climatefever[pure].[split].jsonl > fever-climatefever[pure].[split].jsonl
```
5. Run model training codes, recommended to use GPU.

All BERT and XLNet training codes uses ```mainenv.yml``` environment and are located in ```03_Model_Training/``` with tuned hyperparemeters located at the last cell of "Hyperparameter tuning" section. 

DA uses ```daenv.txt``` environment and can be trained using tuned parameters the following command:
```bash
PYTHONPATH=src python src/script/da/mod_train_da.py src/script/da/parameters/da-[training-data].json models/da-document-models --cuda-device [0 if GPU else -1]
```
If "training-data" is not stored under ```data/```, the parameter file JSON fields "train_data_path" and "validation_data_path" will have to point to the data file location. GloVe embedding was also used and if not stored under ```glove/```, then "model/text_field_embedded/tokens/pretrained_file" field have to point to the "glove.6B.300d.txt.gz" location.

Running all training code will yield a directory structure:
```bash
model/
├── da-document-models/
│   ├── climatefeverpure-da-doc-epoch44.tar.gz
│   ├── fever-climatefeverpure-da-doc-epoch86.tar.gz
│   └── fever-da-doc-epoch95.out.tar.gz
├── document-models/
│   ├── climatefeverpure-bert-base-uncased-doc/
│   ├── climatefeverpure-xlnet-base-cased-doc/
│   ├── fever-bert-base-uncased-doc/
│   ├── fever-climatefeverpure-bert-base-uncased-doc/
│   ├── fever-climatefeverpure-xlnet-base-cased-doc/
│   └── fever-xlnet-base-cased-doc/
├── sentence-models/
│   ├── climatefever-bert-base-uncased-sent/
│   ├── climatefever-xlnet-base-cased-sent/
│   ├── climatefeverpure-bert-base-uncased-sent/
│   ├── climatefeverpure-xlnet-base-cased/
│   ├── fever-bert-base-uncased-sent/
│   ├── fever-climatefever-bert-base-uncased-sent/
│   ├── fever-climatefever-xlnet-base-cased-sent/
│   ├── fever-climatefeverpure-bert-base-uncased-sent/
│   ├── fever-climatefeverpure-xlnet-base-cased-sent/
│   └── fever-xlnet-base-cased-sent/
```
where model filenames have pattern "\<training-data\>-\<model-checkpoint\>-\<\[document|sentence\]\>".

6. Run prediction scripts.

Prediction on BERT and XLNet models uses ```mainenv.yml``` environment and predicts with __all__ models under a directory with command:

Document models inference
```bash
PYTHONPATH=src src/script/hf_predict.py data/doc-dataset models/document-models predictions/doc --cuda-device [0 if GPU else -1] --document
```

Sentence (micro-verdict) models inference
```bash
PYTHONPATH=src src/script/hf_predict.py data/sent-dataset models/sentence-models predictions/sent --cuda-device [0 if GPU else -1]
```
Append ```--aggregate``` for macro-verdict majority and mean probability aggregation.

Prediction on DA models uses ```daenv.txt``` environment and predicts with __all__models under a directory with command:
```bash
PYTHONPATH=src src/script/da/batch_da_predict.py data/doc-dataset models/da-document-models predictions/doc
```

Running the predictions will yield directory structure:
```bash
ROOT/
├── predictions/
│   ├── doc/
│   │   ├── climatefever/
│   │   ├── climatefeverpure/
│   │   ├── fever/
│   │   ├── scifact/
│   │   ├── scifact_test/
│   │   └── scifactpipeline/
│   └── sent/
│       ├── climatefever/
│       ├── climatefeverpure/
│       ├── fever/
│       ├── scifact/
│       ├── scifact_test/
│       └── scifactpipeline/
```
The levels in directory represents "predictions/<\[doc|sent\]>/<evaluate_on_dataset>/<training_data-model_pair_predictions>".

7. Run evaluation code.

Run ```maincode/00_Metric_Evaluation/ipynb``` to generate the Scorer objects containing evaluation information and yields the following files:
```bash
ROOT/
├── metrics/
│   ├── concatenate_evidences_metrics.pkl
│   ├── sent_macro_verdict_majority_metrics.pkl
│   ├── sent_macro_verdict_meanproba_metrics.pkl
│   └── sent_micro_verdict_metrics.pkl
```

Using the pickle files:
 - ```maincode/01_Metric_Summarise.ipynb``` generates each training_data-model pair classification report and confusion matrix.
 - ```maincode/02_Total_Error.ipynb``` generates training_data-model all error counts
 - ```maincode/03_Model_Confidence.ipynb``` generates model confidence

8. \[Optional\] Prepare for SciFact submission by running command:
```bash
PYTHONPATH=src python src/script/scifact_submission.py /path/to/dataset /path/to/write -p num_workers
```
for document dataset. For sentence dataset, append ```--sentence --conversion-mode [meanproba|majority|hybrid]```. The conversion mode used for this project was "hybrid". 

The result of this project SciFact test dataset was also uploaded to SciFact leaderboard:
| Model | Link |
|-------|------|
|FEVER + Climate-FEVER Pure XLNet Document Model|https://leaderboard.allenai.org/scifact/submission/civdehrpggp8105g9dq0|
|FEVER + Climate-FEVER XLNet Sentence (Hybrid) Model|https://leaderboard.allenai.org/scifact/submission/civeeihuoqbrkfl6suo0|

# Detailed Directory Tree
```bash
ROOT/
##########################  TeX Source Code
├── 7CCSMPRJ-Report/
│   ├── chapters/
│   │   ├── 00_starters/
│   │   ├── 01_introduction/
│   │   ├── 03_background/
│   │   ├── 04_dataset_statistics/
│   │   ├── 05_design/
│   │   ├── 06_results/
│   │   ├── 07_lsep/
│   │   ├── 08_conclusion/
│   │   ├── 98_FinalDBLP.bib
│   │   └── 99_appendix/
│   ├── img/
│   │   ├── 03_background/
│   │   ├── 05_design/
│   │   ├── 06_results/
│   │   └── 99_appendix/
│   └── main.tex
##########################  Processed data
├── data/
│   ├── doc-dataset/
│   │   ├── climatefever.dev.n5.jsonl
│   │   ├── climatefever.test.n5.jsonl
│   │   ├── climatefeverpure.dev.n5.jsonl
│   │   ├── climatefeverpure.test.n5.jsonl
│   │   ├── climatefeverpure.train.n5.jsonl
│   │   ├── fever-climatefeverpure.dev.n5.jsonl
│   │   ├── fever-climatefeverpure.train.n5.jsonl
│   │   ├── fever.dev.n5.jsonl
│   │   ├── fever.test.n5.jsonl
│   │   ├── fever.train.n5.jsonl
│   │   ├── scifact.all.test.n5.jsonl
│   │   ├── scifact_test.test.n6s5.jsonl
│   │   └── scifactpipeline.all.test.n6.s5.jsonl
│   ├── fever-nei-sampled/
│   │   ├── dev.n5.nei.jsonl
│   │   ├── test.n5.nei.jsonl
│   │   └── train.n5.nei.jsonl
│   ├── feverised-climatefever/
│   │   ├── climatefever_corpus.jsonl
│   │   ├── climatefever_paper_all.jsonl
│   │   ├── climatefever_paper_all_titleid.jsonl
│   │   ├── dev.climatefeverpure.jsonl
│   │   ├── lineid_translator.json
│   │   ├── test.climatefeverpure.jsonl
│   │   └── train.climatefeverpure.jsonl
│   ├── feverised-climatefever_sent/
│   │   ├── climatefever_corpus.jsonl
│   │   ├── climatefever_paper_all.jsonl
│   │   ├── climatefever_paper_all_titleid.jsonl
│   │   ├── dev.climatefever.jsonl
│   │   ├── lineid_translator.json
│   │   ├── test.climatefever.jsonl
│   │   └── train.climatefever.jsonl
│   ├── feverised-scifact/
│   │   ├── scifact_all.jsonl
│   │   ├── scifact_all_test.jsonl
│   │   ├── scifact_corpus.jsonl
│   │   ├── scifact_dev.jsonl
│   │   ├── scifact_test.jsonl
│   │   └── scifact_train.jsonl
│   ├── scifact-nei-sampled/
│   │   └── all.n5.nei.jsonl
│   ├── scifact-pipeline-sampled/
│   │   └── scifact.pipeline.test.n6.jsonl
│   ├── scifact-test-sampled/
│   │   └── scifact.test.n6.s5.jsonl
│   └── sent-dataset/
│       ├── climatefever.dev.n5.jsonl
│       ├── climatefever.test.n5.jsonl
│       ├── climatefever.train.n5.jsonl
│       ├── climatefeverpure.dev.n5.jsonl
│       ├── climatefeverpure.test.n5.jsonl
│       ├── climatefeverpure.train.n5.jsonl
│       ├── fever-climatefever.dev.n5.jsonl
│       ├── fever-climatefever.train.n5.jsonl
│       ├── fever-climatefeverpure.dev.n5.jsonl
│       ├── fever-climatefeverpure.train.n5.jsonl
│       ├── fever.dev.n5.jsonl
│       ├── fever.test.n5.jsonl
│       ├── fever.train.n5.jsonl
│       ├── scifact.all.test.n5.jsonl
│       ├── scifact_test.test.n6s5.jsonl
│       └── scifactpipeline.all.test.n6.s5.jsonl
##########################  Evidence corpus databases
├── evidence_corpus_dbs/
│   ├── fever.db
│   ├── feverised-climatefever-titleid.db
│   ├── feverised-scifact.db
│   └── README.txt
##########################  Evaluation output
├── metrics/
│   ├── concatenate_evidences_metrics.pkl
│   ├── sent_macro_verdict_majority_metrics.pkl
│   ├── sent_macro_verdict_meanproba_metrics.pkl
│   └── sent_micro_verdict_metrics.pkl
├── submissions/
│   └── scifact_test/
│       ├── doc/
│       └── hybrid/
##########################  Glove embedding for DA training
├── glove/
│   └── glove.6B.300d.txt.gz
##########################  Trained Models
├── models/
│   ├── da-document-models/
│   │   ├── climatefeverpure-da-doc-epoch44.tar.gz
│   │   ├── fever-climatefeverpure-da-doc-epoch86.tar.gz
│   │   └── fever-da-doc-epoch95.out.tar.gz
│   ├── document-models/
│   │   ├── climatefeverpure-bert-base-uncased-doc/
│   │   ├── climatefeverpure-xlnet-base-cased-doc/
│   │   ├── fever-bert-base-uncased-doc/
│   │   ├── fever-climatefeverpure-bert-base-uncased-doc/
│   │   ├── fever-climatefeverpure-xlnet-base-cased-doc/
│   │   └── fever-xlnet-base-cased-doc/
│   └── sentence-models/
│       ├── climatefever-bert-base-uncased-sent/
│       ├── climatefever-xlnet-base-cased-sent/
│       ├── climatefeverpure-bert-base-uncased-sent/
│       ├── climatefeverpure-xlnet-base-cased/
│       ├── fever-bert-base-uncased-sent/
│       ├── fever-climatefever-bert-base-uncased-sent/
│       ├── fever-climatefever-xlnet-base-cased-sent/
│       ├── fever-climatefeverpure-bert-base-uncased-sent/
│       ├── fever-climatefeverpure-xlnet-base-cased-sent/
│       └── fever-xlnet-base-cased-sent/
##########################  Model predictions
├── predictions/
│   ├── doc/
│   │   ├── climatefever/
│   │   ├── climatefeverpure/
│   │   ├── fever/
│   │   ├── scifact/
│   │   ├── scifact_test/
│   │   └── scifactpipeline/
│   └── sent/
│       ├── climatefever/
│       ├── climatefeverpure/
│       ├── fever/
│       ├── scifact/
│       ├── scifact_test/
│       └── scifactpipeline/
##########################  Source Code
├── maincode/
│   ├── 00_Feverise.ipynb
│   ├── 01_Split_Climate-FEVER.ipynb
│   ├── 03_Model_Training/
│   └── 04_Model_Evaluation/
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── feverise/
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   ├── build_db.py
│   │   ├── build_db_mod.py
│   │   ├── climatefever.py
│   │   ├── climatefever_sent.py
│   │   ├── scifact.py
│   │   └── util.py
│   ├── gen/
│   │   ├── __init__.py
│   │   ├── special.py
│   │   └── util.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── doc_db.py
│   │   ├── fever_doc_db.py
│   │   └── retrieval.py
│   ├── rte/
│   │   ├── aggregate.py
│   │   └── da/
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── chart.py
│   │   └── scorer.py
│   ├── script/
│   │   ├── __init__.py
│   │   ├── batch_prepare_docsent.sh
│   │   ├── da/
│   │   ├── evidence_sampling.py
│   │   ├── hf_predict.py
│   │   ├── prepare_docsent_data.py
│   │   └── scifact_submission.py
│   └── submission/
│       └── scifact.py
##########################  Anaconda/Pip Environment
├── daenv.txt
└── mainenv.yml
```