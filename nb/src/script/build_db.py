#!/usr/bin/env python3

#Reference: https://github.com/sheffieldnlp/naacl2018-fever/blob/master/src/scripts/build_db.py

#Adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import unicodedata

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from rte.da.util_log_helper import LogHelper

LogHelper.setup()
logger = LogHelper.get_logger("DrQA BuildDB")


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    def normalize(text):
        """Resolve different type of unicode encodings."""
        return unicodedata.normalize('NFD', text)
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((normalize(doc['id']), doc['text'], doc['lines']))
    return documents


def store_contents(data_path, save_path, num_workers=None):
    """store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text, lines);")

    workers = ProcessPool(num_workers)
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            c.executemany("INSERT INTO documents VALUES (?,?,?)", pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------
def run(data_path, save_path, num_workers):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        print("Save directory doesn't exist. Making {0}".format(save_dir))
        os.makedirs(save_dir)

    store_contents(
        data_path, save_path, num_workers
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        logger.info("Save directory doesn't exist. Making {0}".format(save_dir))
        os.makedirs(save_dir)

    store_contents(
        args.data_path, args.save_path, args.num_workers
    )
