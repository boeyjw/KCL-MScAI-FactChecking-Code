#!/bin/bash

# NEI Sampling
# SciFact
echo "NEI Sampling SciFact..."
PYTHONPATH=src python src/script/evidence_sampling.py data/feverised-scifact/scifact_all.jsonl data/scifact-nei-sampled/all.n5.nei.jsonl data/feverised-scifact/feverised-scifact.db -p 20 -i beir-v1.0.0-scifact-flat --max-pages 5 --max-sentences 2

# SciFact Pipeline
echo "NEI Sampling SciFact Pipeline..."
PYTHONPATH=src python src/script/evidence_sampling.py data/feverised-scifact/scifact_all_test.jsonl data/scifact-pipeline-sampled/scifact.pipeline.test.n6.jsonl data/feverised-scifact/feverised-scifact.db -p 10 -i beir-v1.0.0-scifact-flat --max-pages 6 --max-sentences 5 --pipeline-mode

# SciFact Test Pipeline
echo "NEI Sampling SciFact Test Pipeline..."
PYTHONPATH=src python src/script/evidence_sampling.py data/feverised-scifact/scifact_test.jsonl data/scifact-test-sampled/scifact.test.n6.s5.jsonl data/feverised-scifact/feverised-scifact.db -p 10 -i beir-v1.0.0-scifact-flat --max-pages 6 --max-sentences 5 --pipeline-mode

# FEVER
echo "NEI Sampling FEVER Dev..."
PYTHONPATH=src python src/script/evidence_sampling.py data/fever/paper_dev.jsonl data/fever-nei-sampled/dev.n5.nei.jsonl data/fever/fever.db -p 10 -i beir-v1.0.0-fever-flat --max-pages 5 --max-sentences 2

echo "NEI Sampling FEVER Test..."
PYTHONPATH=src python src/script/evidence_sampling.py data/fever/paper_test.jsonl data/fever-nei-sampled/test.n5.nei.jsonl data/fever/fever.db -p 10 -i beir-v1.0.0-fever-flat --max-pages 5 --max-sentences 2

echo "NEI Sampling FEVER Train..."
PYTHONPATH=src python src/script/evidence_sampling.py data/fever/train.jsonl data/fever-nei-sampled/train.n5.nei.jsonl /users/k21190024/study/fact-check-transfer-learning/scratch/data/fever/fever.db -p 10 -i beir-v1.0.0-fever-flat --max-pages 5 --max-sentences 2

# Prepare document datasets
# Climate-FEVER Pure
echo "Prepare DocData Climate-FEVER Pure..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever/climatefever_dev.jsonl data/doc-dataset/climatefeverpure.dev.n5.jsonl data/feverised-climatefever/feverised-climatefever-titleid.db -p 10 --id-prefix climatefeverpure

PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever/climatefever_test.jsonl data/doc-dataset/climatefeverpure.test.n5.jsonl data/feverised-climatefever/feverised-climatefever-titleid.db -p 10 --id-prefix climatefeverpure

PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever/climatefever_train.jsonl data/doc-dataset/climatefeverpure.train.n5.jsonl data/feverised-climatefever/feverised-climatefever-titleid.db -p 10 --id-prefix climatefeverpure

# Climate-FEVER
echo "Prepare DocData Climate-FEVER..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever_sent/climatefever_dev.jsonl data/doc-dataset/climatefever.dev.n5.jsonl data/feverised-climatefever_sent/feverised-climatefever-titleid.db -p 10 --id-prefix climatefever

PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever_sent/climatefever_test.jsonl data/doc-dataset/climatefever.test.n5.jsonl data/feverised-climatefever_sent/feverised-climatefever-titleid.db -p 10 --id-prefix climatefever

# SciFact
echo "Prepare DocData SciFact..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/scifact-nei-sampled/all.n5.nei.jsonl data/doc-dataset/scifact.all.n5.jsonl data/feverised-scifact/feverised-scifact.db -p 10

# SciFact Pipeline
echo "Prepare DocData SciFact Pipeline..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/scifact-pipeline-sampled/scifact.pipeline.test.n6.jsonl data/doc-dataset/scifactpipeline.all.test.n6.s5.jsonl data/feverised-scifact/feverised-scifact.db -p 10 --pipeline-mode

# SciFact Test Pipeline
echo "Prepare DocData SciFact Test Pipeline..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/scifact-test-sampled/scifact.test.n6.s5.jsonl data/doc-dataset/scifact_test.test.n6s5.jsonl data/feverised-scifact/feverised-scifact.db -p 10 --pipeline-mode

# FEVER
echo "Prepare DocData FEVER..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/fever-nei-sampled/dev.n5.nei.jsonl data/doc-dataset/fever.dev.n5.jsonl data/fever/fever.db -p 10

PYTHONPATH=src python src/script/prepare_docsent_data.py data/fever-nei-sampled/test.n5.nei.jsonl data/doc-dataset/fever.test.n5.jsonl data/fever/fever.db -p 10

PYTHONPATH=src python src/script/prepare_docsent_data.py data/fever-nei-sampled/train.n5.nei.jsonl data/doc-dataset/fever.train.n5.jsonl data/fever/fever.db -p 10

# FEVER + Climate-FEVER Pure
echo "Prepare DocData FEVER + Climate-FEVER Pure..."
cat data/doc-dataset/fever.train.n5.jsonl data/doc-dataset/climatefeverpure.train.n5.jsonl > data/doc-dataset/fever-climatefeverpure.train.n5.jsonl

cat data/doc-dataset/fever.dev.n5.jsonl data/doc-dataset/climatefeverpure.dev.n5.jsonl > data/doc-dataset/fever-climatefeverpure.dev.n5.jsonl

# Prepare sentence datasets
# Climate-FEVER Pure
echo "Prepare SentData Climate-FEVER Pure..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever/climatefever_dev.jsonl data/sent-dataset/climatefeverpure.dev.n5.jsonl data/feverised-climatefever/feverised-climatefever-titleid.db -p 10 --sentence-pair --id-prefix climatefeverpure

PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever/climatefever_test.jsonl data/sent-dataset/climatefeverpure.test.n5.jsonl data/feverised-climatefever/feverised-climatefever-titleid.db -p 10 --sentence-pair --id-prefix climatefeverpure

PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever/climatefever_train.jsonl data/sent-dataset/climatefeverpure.train.n5.jsonl data/feverised-climatefever/feverised-climatefever-titleid.db -p 10 --sentence-pair --id-prefix climatefeverpure

# Climate-FEVER
echo "Prepare SentData Climate-FEVER..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever_sent/climatefever_dev.jsonl data/sent-dataset/climatefever.dev.n5.jsonl data/feverised-climatefever_sent/feverised-climatefever-titleid.db -p 10 --sentence-pair --id-prefix climatefever

PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever_sent/climatefever_test.jsonl data/sent-dataset/climatefever.test.n5.jsonl data/feverised-climatefever_sent/feverised-climatefever-titleid.db -p 10 --sentence-pair --id-prefix climatefever

PYTHONPATH=src python src/script/prepare_docsent_data.py data/feverised-climatefever_sent/climatefever_train.jsonl data/sent-dataset/climatefever.train.n5.jsonl data/feverised-climatefever_sent/feverised-climatefever-titleid.db -p 10 --sentence-pair --id-prefix climatefever

# SciFact
echo "Prepare SentData SciFact..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/scifact-nei-sampled/all.n5.nei.jsonl data/sent-dataset/scifact.all.n5.jsonl data/feverised-scifact/feverised-scifact.db -p 10 --sentence-pair

# SciFact Pipeline
echo "Prepare SentData SciFact Pipeline..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/scifact-pipeline-sampled/scifact.pipeline.test.n6.jsonl data/sent-dataset/scifactpipeline.all.test.n6.s5.jsonl data/feverised-scifact/feverised-scifact.db -p 10 --sentence-pair --pipeline-mode

# SciFact Test Pipeline
echo "Prepare SentData SciFact Test Pipeline..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/scifact-test-sampled/scifact.test.n6.s5.jsonl data/sent-dataset/scifact_test.test.n6s5.jsonl data/feverised-scifact/feverised-scifact.db -p 10 --sentence-pair --pipeline-mode

# FEVER
echo "Prepare SentData FEVER..."
PYTHONPATH=src python src/script/prepare_docsent_data.py data/fever-nei-sampled/dev.n5.nei.jsonl data/sent-dataset/fever.dev.n5.jsonl data/fever/fever.db -p 10 --sentence-pair

PYTHONPATH=src python src/script/prepare_docsent_data.py data/fever-nei-sampled/test.n5.nei.jsonl data/sent-dataset/fever.test.n5.jsonl data/fever/fever.db -p 10 --sentence-pair

PYTHONPATH=src python src/script/prepare_docsent_data.py data/fever-nei-sampled/train.n5.nei.jsonl data/sent-dataset/fever.train.n5.jsonl data/fever/fever.db -p 10 --sentence-pair --max-evidence 5

# FEVER + Climate-FEVER
echo "Prepare SentData FEVER + Climate-FEVER..."
cat data/sent-dataset/fever.train.n5.jsonl data/sent-dataset/climatefever.train.n5.jsonl > data/sent-dataset/fever-climatefever.train.n5.jsonl

cat data/sent-dataset/fever.dev.n5.jsonl data/sent-dataset/climatefever.dev.n5.jsonl > data/sent-dataset/fever-climatefever.dev.n5.jsonl

# FEVER + Climate-FEVER Pure
echo "Prepare SentData FEVER + Climate-FEVER Pure..."
cat data/sent-dataset/fever.train.n5.jsonl data/sent-dataset/climatefeverpure.train.n5.jsonl > data/sent-dataset/fever-climatefeverpure.train.n5.jsonl

cat data/sent-dataset/fever.dev.n5.jsonl data/sent-dataset/climatefeverpure.dev.n5.jsonl > data/sent-dataset/fever-climatefeverpure.dev.n5.jsonl