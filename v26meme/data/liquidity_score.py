import math
import pandas as pd
from typing import Dict, Any, List, Tuple

def get_depth5_usd(ob: Dict[str, List[List[float]]]) -> float:
    """Calculate the notional value of the top 5 levels of the order book."""
    bids = ob.get('bids', [])
    asks = ob.get('asks', [])
    if not bids or not asks:
        return 0.0
    bid_depth = sum(float(p) * float(q) for p, q, *_ in bids[:5])
    ask_depth = sum(float(p) * float(q) for p, q, *_ in asks[:5])
    return (bid_depth + ask_depth) / 2.0

def get_spread_bps(ob: Dict[str, List[List[float]]]) -> float:
    """Calculate the bid-ask spread in basis points."""
    bids = ob.get('bids', [])
    asks = ob.get('asks', [])
    if not bids or not asks:
        return float('inf')
    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid <= 0:
        return float('inf')
    return ((best_ask - best_bid) / best_bid) * 10000.0

def get_impact_bps(ob: Dict[str, List[List[float]]], typical_order_usd: float) -> float:
    """Calculate the price impact for a typical order size."""
    bids = ob.get('bids', [])
    asks = ob.get('asks', [])
    if not bids or not asks:
        return float('inf')
    
    mid_price = (bids[0][0] + asks[0][0]) / 2.0
    if mid_price == 0:
        return float('inf')

    def _vwap(levels: List[List[float]], notional: float) -> float:
        filled_qty = 0
        total_cost = 0
        for price, qty, *_ in levels:
            level_notional = price * qty
            if total_cost + level_notional >= notional:
                needed_qty = (notional - total_cost) / price
                filled_qty += needed_qty
                total_cost += needed_qty * price
                break
            else:
                filled_qty += qty
                total_cost += level_notional
        return total_cost / filled_qty if filled_qty > 0 else float('inf')

    buy_vwap = _vwap(asks, typical_order_usd)
    sell_vwap = _vwap(bids, typical_order_usd)
    
    buy_impact = abs((buy_vwap - mid_price) / mid_price) * 10000 if buy_vwap != float('inf') else float('inf')
    sell_impact = abs((sell_vwap - mid_price) / mid_price) * 10000 if sell_vwap != float('inf') else float('inf')
    
    return max(buy_impact, sell_impact)

def calculate_liquidity_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a composite liquidity score from a DataFrame of metrics.
    The DataFrame must contain: vol_24h_usd, depth5_usd, spread_bps, impact_bps.
    """
    df = metrics_df.copy()
    
    # Percentile ranks (higher is better)
    df['r_vol'] = df['vol_24h_usd'].rank(pct=True)
    df['r_depth'] = df['depth5_usd'].rank(pct=True)
    
    # Inverse percentile ranks (lower is better, so we subtract from 1)
    df['r_spread'] = 1 - df['spread_bps'].rank(pct=True)
    df['r_impact'] = 1 - df['impact_bps'].rank(pct=True)
    
    # Composite score
    df['liq_score'] = (
        0.4 * df['r_vol'] +
        0.4 * df['r_depth'] +
        0.1 * df['r_spread'] +
        0.1 * df['r_impact']
    )
    return df
