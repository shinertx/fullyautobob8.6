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
        # Filter alphas that have performance data for the specified regime
        usable = [a for a in active_alphas if a.get('performance', {}).get(regime, {}).get('n_trades', 0) > 5]
        if not usable:
            # Fallback to 'all' regime
            regime = 'all'
            usable = [a for a in active_alphas if a.get('performance', {}).get('all', {}).get('n_trades', 0) > 5]
        if not usable: 
            # If no alphas have sufficient performance data, return empty weights
            return {}

        returns_data = {a['id']: a.get('performance', {}).get(regime, {}).get('returns', []) for a in usable}
        max_len = max(len(v) for v in returns_data.values()) if returns_data else 0
        if max_len == 0:
            # If no returns data available, return equal weights
            equal_weight = 1.0 / len(usable)
            return {a['id']: equal_weight for a in usable}
        
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
