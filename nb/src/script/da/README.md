mod_train_da: PYTHONPATH=src python src/script/da/mod_train_da.py src/script/da/parameters/<parameter-file>.json <dump-path> --cuda-device GPU_ID

mod_eval_da: PYTHONPATH=src python src/script/da/mod_eval_da.py <model-path> <data-to-infer-path> <dump-path>
    
batch_da_predict: PYTHONPATH=src python src/script/da/batch_da_predict.py <document-data-dir> <da-archive-models-dir> <dump-path>