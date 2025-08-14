import random, json, hashlib
import pandas as pd
from typing import List, Dict, Any
from v26meme.labs.simlab import SimLab

class FeatureProber:
    """
    Perturb leaf thresholds/operators to measure robustness.
    Returns robustness score in [0,1].
    """
    def __init__(self, fees_bps: float, slippage_bps: float, perturbations: int = 64, delta_fraction: float = 0.15, seed: int = 7):
        self.simlab = SimLab(fees_bps, slippage_bps)
        self.k = int(perturbations)
        self.d = float(delta_fraction)
        random.seed(seed)

    def _leaves(self, formula):
        # returns list of (node_ref) for in-place mutation
        acc, q = [], [formula]
        while q:
            n = q.pop(0)
            if isinstance(n[0], list):
                q.append(n[0]); q.append(n[2])
            else:
                acc.append(n)
        return acc

    def score(self, df_features: pd.DataFrame, formula: List[Any]) -> Dict[str, Any]:
        base = self.simlab.run_backtest(df_features, formula).get("all", {})
        if not base or base.get("n_trades", 0) == 0:
            return {"robust_score": 0.0, "base": base}
        base_mean = base.get("avg_return", 0.0)
        base_mdd = abs(base.get("mdd", 1.0))
        ok = 0
        for _ in range(self.k):
            import copy
            f2 = copy.deepcopy(formula)
            leaves = self._leaves(f2)
            if not leaves: continue
            leaf = random.choice(leaves)
            if not isinstance(leaf[0], list):
                # Jitter threshold or flip operator with small prob
                if random.random() < 0.8:
                    thr = float(leaf[2])
                    leaf[2] = thr * (1.0 + random.uniform(-self.d, self.d))
                else:
                    leaf[1] = ">" if leaf[1] == "<" else "<"
            res = self.simlab.run_backtest(df_features, f2).get("all", {})
            if not res or res.get("n_trades",0)==0: 
                continue
            # consider robust if sign of mean preserved and drawdown not catastrophically worse
            if (base_mean >= 0 and res.get("avg_return", -1e9) >= 0) or (base_mean < 0 and res.get("avg_return", 1e9) < 0):
                if abs(res.get("mdd", 1.0)) <= base_mdd * 1.25:
                    ok += 1
        return {"robust_score": ok / float(self.k), "base": base}
