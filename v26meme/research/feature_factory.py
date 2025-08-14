import pandas as pd
import numpy as np

class FeatureFactory:
    def create(self, df: pd.DataFrame, symbol: str | None = None,
               cfg: dict | None = None, other_dfs: dict | None = None) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()

        # Core features (PIT-safe)
        out['return_1p'] = out['close'].pct_change()
        out['volatility_20p'] = out['return_1p'].rolling(20).std() * np.sqrt(20)
        out['momentum_10p'] = out['close'].pct_change(10)
        delta = out['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        out['rsi_14'] = 100 - (100 / (1 + rs))
        out['sma_50'] = out['close'].rolling(50).mean()
        out['close_vs_sma50'] = (out['close'] - out['sma_50']) / out['sma_50']

        # Time-of-day (PIT)
        if hasattr(out.index, "hour"):
            out['hod_sin'] = np.sin(2 * np.pi * out.index.hour / 24.0)
            out['hod_cos'] = np.cos(2 * np.pi * out.index.hour / 24.0)
        else:
            out['hod_sin'] = 0.0
            out['hod_cos'] = 0.0

        # Round-number proximity (scale-aware, PIT)
        logp = np.log10(out['close'].clip(lower=1e-9))
        frac = logp - np.floor(logp)
        prox = np.minimum(np.abs(frac), np.abs(1 - frac))
        out['round_proximity'] = -prox

        # Cross-asset (lagged)
        if other_dfs:
            btc = other_dfs.get('BTC_USD_SPOT')
            eth = other_dfs.get('ETH_USD_SPOT')
            if btc is not None and not btc.empty:
                btc_close = btc['close'].reindex(out.index, method='pad')
                out['btc_corr_20p'] = out['close'].rolling(20).corr(btc_close).shift(1)
            if eth is not None and btc is not None and not eth.empty and not btc.empty:
                eth_close = eth['close'].reindex(out.index, method='pad')
                btc_close = btc['close'].reindex(out.index, method='pad')
                out['eth_btc_ratio'] = (eth_close / btc_close).shift(1)

        return out.dropna()
