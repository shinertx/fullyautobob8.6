from typing import Dict, Optional
from loguru import logger

class USDFX:
    """USD converter using observed tickers."""
    def __init__(self, stable_warn_bps: float = 100.0):
        self.stable_warn = stable_warn_bps / 10000.0
        self.cache: Dict[str, float] = {}  # SYMBOL -> USD rate

    def load_from_tickers(self, tickers_by_venue: Dict[str, Dict[str, dict]]):
        self.cache = {"USD": 1.0}
        for _venue, ticks in tickers_by_venue.items():
            for sym, t in ticks.items():
                if "/" not in sym: 
                    continue
                a, b = sym.split("/")
                last = t.get("last") or t.get("close")
                if not last: 
                    continue
                try:
                    p = float(last)
                except Exception:
                    continue
                if b == "USD":
                    self.cache[a] = p
                elif a == "USD" and p != 0:
                    self.cache[b] = 1.0/p

        for stable in ["USDT","USDC","DAI","FDUSD","TUSD","PYUSD"]:
            r = self.cache.get(stable)
            if r is not None and abs(r-1.0) > self.stable_warn:
                logger.warning(f"Stablecoin {stable}/USD parity off by {abs(r-1.0):.2%}")

    def to_usd(self, quote_symbol: str) -> Optional[float]:
        return self.cache.get(quote_symbol, None)
