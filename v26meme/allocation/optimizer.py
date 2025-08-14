import numpy as np, pandas as pd

class PortfolioOptimizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg['portfolio']

    def _inv_var_weights(self, returns_df: pd.DataFrame) -> dict:
        if returns_df.shape[1] == 1:
            return {returns_df.columns[0]: 1.0}
        inv_var = 1 / returns_df.var().replace(0, np.nan)
        inv_var = inv_var.fillna(inv_var.max())
        return (inv_var / inv_var.sum()).to_dict()

    def get_weights(self, active_alphas: list, regime: str) -> dict:
        if not active_alphas: return {}
        usable = [a for a in active_alphas if regime in a['performance'] and a['performance'][regime].get('n_trades',0)>5]
        if not usable:
            regime = 'all'
            usable = [a for a in active_alphas if 'all' in a['performance'] and a['performance']['all'].get('n_trades',0)>5]
        if not usable: return {}

        returns_data = {a['id']: a['performance'][regime].get('returns', []) for a in usable}
        max_len = max(len(v) for v in returns_data.values())
        for k, v in returns_data.items():
            v.extend([0.0]*(max_len - len(v)))
        df = pd.DataFrame(returns_data)
        weights = self._inv_var_weights(df)

        # caps & floor
        for k, w in list(weights.items()):
            if w > self.cfg['max_alpha_concentration']: weights[k] = self.cfg['max_alpha_concentration']
            if w < self.cfg['min_allocation_weight']: weights[k] = 0.0
        tot = sum(weights.values())
        return {k: (w/tot) for k, w in weights.items()} if tot>0 else {}
