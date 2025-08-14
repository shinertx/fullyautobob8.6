from typing import List, Dict, Any, Tuple
from loguru import logger
import ccxt, time

from v26meme.data.asset_registry import make_instrument
from v26meme.data.usd_fx import USDFX

def _spread_bps(t: dict) -> float:
    bid = t.get('bid'); ask = t.get('ask')
    if not bid or not ask or bid <= 0 or ask <= 0: return float('inf')
    mid = 0.5 * (bid + ask)
    return 10000.0 * (ask - bid) / mid if mid > 0 else float('inf')

def _impact_bps(ex, market_id: str, notional_usd: float, usd_per_quote: float) -> float:
    try:
        ob = ex.fetch_order_book(market_id, limit=10)
    except Exception as e:
        logger.warning(f"Could not fetch order book for {market_id} on {ex.id}: {e}")
        return float('inf')
    asks = ob.get('asks') or []
    bids = ob.get('bids') or []
    if not asks or not bids: return float('inf')
    need_quote = notional_usd / max(1e-12, usd_per_quote)
    filled = 0.0; cost = 0.0
    for price, qty, *_ in asks:
        take = min(qty, need_quote - filled)
        if take <= 0: break
        cost += take * price
        filled += take
        if filled >= need_quote: break
    if filled < need_quote: return float('inf')
    vwap = cost / filled
    best_bid, *_ = bids[0]
    best_ask, *_ = asks[0]
    mid = 0.5*(best_bid + best_ask)
    return 10000.0 * (vwap - mid) / mid if mid>0 else float('inf')

class UniverseScreener:
    def __init__(self, exchanges: List[str], screener_cfg: dict, feeds_cfg: dict | None = None):
        self.exchanges = {}
        for ex in exchanges:
            try:
                obj = getattr(ccxt, ex)()
                obj.load_markets()
                self.exchanges[ex] = obj
            except Exception as e:
                logger.warning(f"Could not init {ex}: {e}")
        self.cfg = screener_cfg
        self.feeds = feeds_cfg or {}

    def _apply_sentiment_boost(self, candidates: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        weight = float(self.cfg.get('sentiment_weight', 0))
        if weight <= 0: return candidates
        cp_cfg = (self.feeds.get('cryptopanic') or {})
        if not cp_cfg.get('enabled'): return candidates
        try:
            from v26meme.feeds.cryptopanic import CryptoPanicFeed
        except Exception:
            return candidates
        feed = CryptoPanicFeed(cp_cfg.get('window_hours', 6), cp_cfg.get('min_score', -1.0))
        bases = list({c[0]['base'] for c in candidates})
        scores = feed.scores_by_ticker(bases)
        boosted = []
        for (inst, vol) in candidates:
            s = scores.get(inst['base']) or []
            if not s:
                boosted.append((inst, vol)); continue
            m = sum(d['score'] for d in s[-5:]) / max(1, len(s[-5:]))
            mult = max(0.9, min(1.1, 1.0 + weight * 0.5 * m))
            boosted.append((inst, vol * mult))
        return boosted

    def get_active_universe(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, dict]]]:
        if not self.exchanges: return [], {}
        tickers_by_venue: Dict[str, Dict[str, dict]] = {}
        for name, ex in self.exchanges.items():
            try: tickers_by_venue[name] = ex.fetch_tickers()
            except Exception as e: 
                logger.error(f"fetch_tickers failed on {name}: {e}")

        if not tickers_by_venue:
            logger.error("No tickers fetched; screener empty.")
            return [], {}

        fx = USDFX(self.cfg.get('stablecoin_parity_warn_bps', 100))
        fx.load_from_tickers(tickers_by_venue)

        candidates: List[Tuple[Dict[str, Any], float]] = []
        for venue, ex in self.exchanges.items():
            markets = ex.markets or {}
            ticks = tickers_by_venue.get(venue, {})
            for sym, m in markets.items():
                if (m.get("swap") or m.get("future")) and not self.cfg.get('derivatives_enabled', False):
                    continue
                if "/" not in sym: 
                    continue
                t = ticks.get(sym, {})
                last = t.get('last') or t.get('close')
                if not last: 
                    continue
                try: price = float(last)
                except Exception: continue

                quote = m.get("quote")
                if quote not in ("USD","USDT","USDC","DAI","FDUSD","TUSD","PYUSD"): continue
                rate = 1.0 if quote == "USD" else fx.to_usd(quote)
                if rate is None: continue

                qv = t.get('quoteVolume'); bv = t.get('baseVolume')
                try:
                    vol_usd = (float(qv) * float(rate)) if qv else (float(bv) * float(price)) if bv else 0.0
                except Exception:
                    vol_usd = 0.0
                if vol_usd < self.cfg['min_24h_volume_usd'] or price < self.cfg['min_price']: continue

                spr = _spread_bps(t)
                if spr == float('inf') or spr > self.cfg['max_spread_bps']: continue

                inst = make_instrument(venue, m)
                inst_dict = {
                    "venue": inst.venue, "type": inst.type, "market_id": inst.market_id,
                    "base": inst.base.symbol, "quote": inst.quote.symbol,
                    "precision": inst.precision, "limits": inst.limits,
                    "display": inst.display, "spread_bps": spr, "price": price,
                    "volume_24h_usd": vol_usd, "usd_per_quote": rate
                }
                candidates.append((inst_dict, vol_usd))

        if not candidates:
            logger.warning("Screener filters removed all markets.")
            return [], tickers_by_venue

        candidates = self._apply_sentiment_boost(candidates)

        short = sorted(candidates, key=lambda kv: kv[1], reverse=True)[: self.cfg['max_markets'] * 3]
        selected: List[Tuple[Dict[str, Any], float]] = []
        for (inst, vol) in short:
            ex = self.exchanges[inst['venue']]
            try:
                imp = _impact_bps(ex, inst['market_id'], self.cfg['typical_order_usd'], inst.get('usd_per_quote',1.0))
            except Exception:
                imp = float('inf')
            time.sleep(max(0.0, getattr(ex, 'rateLimit', 100)/1000.0))
            if imp == float('inf') or imp > self.cfg['max_impact_bps']: continue
            inst['impact_bps'] = imp
            selected.append((inst, vol))

        if not selected:
            logger.warning("Impact screening removed all markets.")
            return [], tickers_by_venue

        selected.sort(key=lambda kv: kv[1], reverse=True)
        return [k for k,_ in selected[: self.cfg['max_markets']]], tickers_by_venue
