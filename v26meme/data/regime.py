import pandas as pd

def label_regime(df: pd.DataFrame, vol_window=20, trend_window=20) -> pd.Series:
    if df.empty or len(df) < max(vol_window, trend_window):
        return pd.Series(["chop"] * len(df), index=df.index)
    close = df['close']
    returns = close.pct_change()
    vol = returns.rolling(vol_window).std()
    regimes = pd.Series("chop", index=df.index)
    regimes[vol > vol.quantile(0.75)] = "high_vol"
    trend = close.rolling(trend_window).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] if x.iloc[0] != 0 else 1.0), raw=False)
    regimes[(trend > 0.05) & (regimes == "chop")] = "trend_up"
    regimes[(trend < -0.05) & (regimes == "chop")] = "trend_down"
    return regimes
