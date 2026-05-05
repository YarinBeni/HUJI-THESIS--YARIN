"""
05_compute_cls_tfidf.py — Classification probe for TF-IDF baseline.
Runs locally (no GPU / no large activations needed).

Output: results/orcc_round1/cls/cls_results_tfidf.json
"""

import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR)) if str(_THIS_DIR) not in __import__('sys').path else None

import sys
sys.path.insert(0, str(_THIS_DIR))

from cls_utils import fit_cls_cv

ORCC_PARQUET = _THIS_DIR / '../../data/evaluation/corpora/orcc_corpus.parquet'
SEAL_PARQUET = _THIS_DIR / '../../data/evaluation/corpora/seal_corpus.parquet'
OUT_DIR      = _THIS_DIR / 'results' / 'orcc_round1' / 'cls'

MIN_COUNT = 5
CLEANINGS = ['tier0', 'maximal']
TEXT_COL  = {'tier0': 'text_tier0', 'maximal': 'text_maximal'}


def main():
    seal_df = pd.read_parquet(SEAL_PARQUET)
    orcc_df = pd.read_parquet(ORCC_PARQUET)

    labeled_mask     = ~orcc_df['year'].isna()
    labeled_orcc_idx = np.where(labeled_mask)[0]
    orcc_sub         = orcc_df.iloc[labeled_orcc_idx]

    task_labels = {}
    for task, col in [('ruler', 'ruler'), ('year', 'year')]:
        raw    = orcc_sub[col].astype(str).values
        counts = pd.Series(raw).value_counts()
        keep   = counts[counts >= MIN_COUNT].index
        mask   = np.isin(raw, keep)
        task_labels[task] = {
            'y':    raw[mask],
            'mask': mask,
            'n_dropped': int((~mask).sum()),
            'n_classes': int(keep.shape[0]),
        }
        print(f"Task '{task}': {mask.sum()} fragments, "
              f"{task_labels[task]['n_classes']} classes "
              f"(dropped {task_labels[task]['n_dropped']} with <{MIN_COUNT})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / 'cls_results_tfidf.json'
    results: dict = json.load(open(results_path)) if results_path.exists() else {}

    t_start = time.time()
    for cleaning in CLEANINGS:
        col = TEXT_COL[cleaning]
        texts = orcc_sub[col].fillna('').tolist()

        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
        X_all = vec.fit_transform(texts)  # sparse (n_labeled, vocab)

        for task, info in task_labels.items():
            X_task = normalize(X_all[info['mask']], norm='l2')
            # Convert to dense for LogisticRegression (vocab is large but n is small)
            X_dense = X_task.toarray()
            m = fit_cls_cv(X_dense, info['y'], cv_strategy='stratified', n_splits=5)
            config_key = f'tfidf__{cleaning}__na__L00__{task}'
            results[config_key] = {
                'method': 'tfidf', 'cleaning': cleaning, 'pooling': 'na',
                'layer': 0, 'task': task, 'n_dropped': info['n_dropped'], **m,
            }
            print(f"  {cleaning}/{task}: acc={m['accuracy_mean']:.3f}  "
                  f"macro_f1={m['macro_f1_mean']:.3f}  "
                  f"chance_acc={m['chance_accuracy']:.3f}")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} configs → {results_path}")
    print(f"Wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
