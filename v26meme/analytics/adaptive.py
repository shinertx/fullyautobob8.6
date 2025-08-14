import numpy as np
import pandas as pd
from typing import Dict, Any
from v26meme.core.state import StateManager

def publish_adaptive_knobs(state: StateManager, cfg: Dict[str, Any], btc_df: pd.DataFrame):
    acfg = cfg.get('adaptive', {}) or {}
    if not acfg.get('enabled', True):
        return
    # Dynamic daily stop: scaled to trailing realized vol on BTC
    if btc_df is not None and not btc_df.empty and 'close' in btc_df:
        rets = btc_df['close'].pct_change().dropna()
        if len(rets) > 10:
            window = int(acfg.get('stop_vol_window_bars', 24))
            vol = rets.rolling(window).std().iloc[-1] if len(rets) >= window else rets.std()
            # Map vol -> stop in [floor, ceil]
            floor = float(acfg.get('daily_stop_pct_floor', 0.05))
            ceil  = float(acfg.get('daily_stop_pct_ceiling', 0.20))
            # simple mapping: 6x daily vol (heuristic), clamped
            target = 6.0 * float(vol)
            target = max(floor, min(ceil, target))
            state.set('adaptive:daily_stop_pct', target)

    # Screener size adaptive: tie to promotion yield (last N cycles) and loop time budget
    # Keep simple: interpolate to mid range for now (safe default); can be extended.
    s_min = int(acfg.get('screener_max_markets_min', 12))
    s_max = int(acfg.get('screener_max_markets_max', 36))
    state.set('adaptive:screener_max_markets', int(0.5*(s_min + s_max)))

    # Population size adaptive: keep between min/max for deterministic compute
    p_min = int(acfg.get('population_size_min', 120))
    p_max = int(acfg.get('population_size_max', 240))
    state.set('adaptive:population_size', int(0.5*(p_min + p_max)))
