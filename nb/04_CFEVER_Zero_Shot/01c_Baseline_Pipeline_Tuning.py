import os
import sys
import subprocess
import shlex

if __name__ == "__main__":
    os.putenv("PYTHONPATH", os.pathsep.join([os.getenv("PYTHONPATH",""),"/users/k21190024/study/fact-checking-repos/fever/baseline/src"]))
    
    pages = range(5, 11)
    sents = range(4, 11)
    splits = ["train", "dev"]
    models = {
        "extend-vocab": "/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/zeroshot/finetune/baseline/rte/extend-vocab_model-9epoch/edited_model.tar.gz",
        "keep-vocab": "/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/zeroshot/finetune/baseline/rte/keep-vocab_model-15epoch/edited_model.tar.gz"
    }
    ir_path = "/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/zeroshot/finetune/baseline/ir"
    dump_path = "/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/zeroshot/finetune/baseline/pipeline"
    
    # 1 - model; 2 - IR data; 3 - predicted labels
    predict_command = "{0} -m allennlp.run predict " \
                    "{1} " \
                    "{2} " \
                    "--output-file {3} " \
                    "--predictor fever " \
                    "--include-package fever.reader " \
                    "--cuda-device 0 " \
                    "--silent"
    submission_command = "{0} -m fever.submission.prepare " \
                    "--predicted_labels {1} " \
                    "--predicted_evidence {2} " \
                    "--out_file {3}"
    
    # cannot parallelise because for unknown reasons
    # path strings turn to object at<xxxxx when 1 loop of Popen has been
    # successfully executed
    for p in pages:
        for s in sents:
            # predict
            procs = []
            for k, model_path in models.items():
                cur_dump = os.path.join(dump_path, k, "p{0}s{1}".format(p, s))
                os.mkdir(cur_dump)
                
                for split in splits:
                    cur_ir = os.path.join(ir_path, split, "{0}.sentences.p{1}.s{2}.jsonl".format(split, p, s))
                    cur_labels = os.path.join(cur_dump, "labels.{0}.jsonl".format(split))
                    out_file = os.path.join(dump_path, k, "p{0}s{1}".format(p, s), "pred.{0}.jsonl".format(split))
                    
                    _ = subprocess.run(
                        shlex.split(predict_command.format(sys.executable, model_path, cur_ir, cur_labels)), 
                        cwd="/users/k21190024/study/fact-check-transfer-learning/repos/fever/baseline",
                        stdout=subprocess.DEVNULL,
                        check=True
                    )
                    _ = subprocess.run(
                        shlex.split(submission_command.format(sys.executable, cur_labels, cur_ir, out_file)), 
                        cwd="/users/k21190024/study/fact-check-transfer-learning/repos/fever/baseline",
                        stdout=subprocess.DEVNULL,
                        check=True
                    )
