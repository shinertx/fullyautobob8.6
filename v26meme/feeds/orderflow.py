from typing import Dict, Any, List
from loguru import logger

class OrderflowSnap:
    """
    Lightweight live order-book features (no historical backfill).
    Intended for real-time gating / realism checks, not backtests.
    """
    def __init__(self, top_levels: int = 5):
        self.top = int(top_levels)

    def _features_from_ob(self, ob: Dict[str, Any]) -> Dict[str, float]:
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks: 
            return {"of_spread": float("inf")}
        best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
        mid = 0.5 * (best_bid + best_ask)
        spread = (best_ask - best_bid) / mid if mid>0 else float("inf")
        tb = sum(float(b[1]) for b in bids[:self.top])
        ta = sum(float(a[1]) for a in asks[:self.top])
        depth = tb + ta
        imb = (tb - ta) / depth if depth>0 else 0.0
        micro = (best_bid * ta + best_ask * tb) / (ta + tb) if (ta+tb)>0 else mid
        return {
            "of_spread": spread,
            "of_depth": depth,
            "of_imbalance": imb,
            "of_microprice_dev": (micro - mid) / mid if mid>0 else 0.0
        }

    def collect_for_instruments(self, exchange, instruments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for inst in instruments:
            try:
                ob = exchange.fetch_order_book(inst["market_id"], limit=self.top)
                out[inst["display"]] = self._features_from_ob(ob)
            except Exception as e:
                logger.debug(f"orderflow snapshot failed for {inst.get('market_id')}: {e}")
        return out
