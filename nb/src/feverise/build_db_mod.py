#!/usr/bin/env python3

#Adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""
# COMMENT: Modified script from Sheffield-NLP to enable fast query for ID <=> titled_ID

import argparse
import sqlite3
import json
import os
import unicodedata

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm

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
            if not doc:
                continue
            # Add the document
            documents.append((normalize(doc['id']), doc["original_id"], doc['text'], doc['lines']))
    return documents


def store_contents(data_path, save_path, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    print('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id, original_id PRIMARY KEY, text, lines);")

    workers = ProcessPool(num_workers)
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            c.executemany("INSERT INTO documents VALUES (?,?,?,?)", pairs)
            pbar.update()
    print('Read %d docs.' % count)
    print('Committing...')
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
