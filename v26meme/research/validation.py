import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import ttest_1samp

def purged_kfold_indices(n: int, k: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds = []
    fold_size = max(1, n // k)
    for i in range(k):
        start = i * fold_size
        end = n if i == k-1 else min(n, (i+1)*fold_size)
        test_idx = np.arange(start, end)
        train_mask = np.ones(n, dtype=bool)
        emb_lo = max(0, start - embargo)
        emb_hi = min(n, end + embargo)
        train_mask[emb_lo:emb_hi] = False
        folds.append((np.where(train_mask)[0], test_idx))
    return folds

def benjamini_hochberg(pvals: List[float], alpha: float):
    if not pvals:
        return [], 0.0
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = 0.0
    accept = [False]*m
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank/m)*alpha:
            accept[idx] = True
            thresh = max(thresh, pvals[idx])
    return accept, thresh

def panel_cv_stats(panel_returns: Dict[str, pd.Series], k_folds: int, embargo: int, alpha_fdr: float):
    all_oos = []
    for _sym, s in panel_returns.items():
        s = s.dropna()
        if s.empty or len(s) < max(20, k_folds*5):
            continue
        n = len(s)
        folds = purged_kfold_indices(n, k_folds, embargo)
        oos_chunks = [s.iloc[test_idx] for _, test_idx in folds]
        if oos_chunks:
            oos = pd.concat(oos_chunks)
            all_oos.append(oos)
    if not all_oos:
        return {"p_value": 1.0, "mean_oos": 0.0, "n": 0}
    all_oos_concat = pd.concat(all_oos)
    if all_oos_concat.std(ddof=1) == 0 or len(all_oos_concat) < 10:
        p = 1.0
    else:
        _, p = ttest_1samp(all_oos_concat, popmean=0.0, alternative='greater')
        p = float(p) if p is not None else 1.0
    return {"p_value": p, "mean_oos": float(all_oos_concat.mean()), "n": int(all_oos_concat.shape[0])}
