import os
import subprocess

if __name__ == "__main__":
    script = ["python", "/users/k21190024/study/fact-checking-repos/fever/baseline/src/scripts/retrieval/ir.py"]
    args = "--db /users/k21190024/study/fact-checking-repos/fever/baseline/fever2-sample/data/fever-data/fever.db " \
        "--model /users/k21190024/study/fact-checking-repos/fever/baseline/fever2-sample/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz " \
        "--in-file /users/k21190024/study/fact-checking-repos/fever/baseline/fever2-sample/data/fever-data/{0}.jsonl " \
        "--out-file /users/k21190024/study/fact-checking-repos/fever/baseline/thesis/baseline-fever-ir/{1}.sentences.p{2}.s{3}.jsonl " \
        "--max-page {4} " \
        "--max-sent {5}"
    
    os.putenv("PYTHONPATH", os.pathsep.join([os.getenv("PYTHONPATH",""),"/users/k21190024/study/fact-checking-repos/fever/baseline/src"]))
    # datasets = ["train", "dev"]
    # maxp = range(1, 11)
    maxs = range(1, 11)

    # for data in datasets:
    #     for pages in maxp:
    #         if os.path.exists("/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/baseline-fever-ir/{0}.sentences.p{1}.s{2}.jsonl".format(data, pages, 1)):
    #             break
    #         procs = [
    #             subprocess.Popen(
    #                 script + args.format(data, data, pages, sentences, pages, sentences).split(" "), 
    #                 cwd="/users/k21190024/study/fact-check-transfer-learning/repos/fever/baseline", 
    #                 stdout=subprocess.DEVNULL
    #             ) for sentences in maxs 
    #         ]
    #         for p in procs:
    #             p.wait()
    
    pages = 6
    data = "dev"
    if os.path.exists("/users/k21190024/study/fact-checking-repos/fever/baseline/thesis/baseline-fever-ir/{0}.sentences.p{1}.s{2}.jsonl".format(data, pages, 1)):
        raise FileExistsError("{0}_p{1} already done".format(data, pages))
    procs = [
        subprocess.Popen(
            script + args.format(data, data, pages, sentences, pages, sentences).split(" "), 
            cwd="/users/k21190024/study/fact-check-transfer-learning/repos/fever/baseline", 
            stdout=subprocess.DEVNULL
        ) for sentences in maxs 
    ]
    for p in procs:
        p.wait()
