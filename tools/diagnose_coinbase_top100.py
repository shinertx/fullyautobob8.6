#!/usr/bin/env python3
"""
Diagnose Coinbase screener eligibility for top-100 markets by 24h USD volume.

- Fetch all tickers (Coinbase)
- Compute USD conversion via USDFX
- For each spot symbol, compute:
  price, bid/ask, spread_bps, quoteVolume, vol_usd, impact_bps (buy/sell/worst)
  precision/limits (min_cost/min_amount), usd_per_quote
- Evaluate against screener thresholds from configs/config.yaml
- Output a CSV report and console summary with rejection reasons

Paper-safe (read-only). Respects exchange rateLimit.
"""
from __future__ import annotations

import os
import sys
import csv
import time
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, cast

import yaml
import ccxt  # type: ignore

from v26meme.data.usd_fx import USDFX


def sleep_rate_limit(ex, factor: float = 1.15):
    rl = getattr(ex, 'rateLimit', 1000) or 1000
    time.sleep(max(0.001, rl/1000.0 * factor))


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def get_usd_per_quote(ex, quote: Optional[str]) -> Optional[float]:
    if not quote or quote.upper() == 'USD':
        return 1.0
    q = quote.upper()
    for pair in (f"{q}/USD", f"USD/{q}"):
        try:
            if pair in ex.markets:
                t = ex.fetch_ticker(pair)
                sleep_rate_limit(ex)
                if pair.startswith(f"{q}/USD"):
                    return safe_float(t.get('last') or t.get('close')) or 1.0
                else:
                    r = safe_float(t.get('last') or t.get('close'))
                    return (1.0/ r) if r else None
        except Exception:
            continue
    return None


def side_impact_bps(side: str, mid: float, depth: List[Tuple[float,float]], order_quote: float) -> float:
    if mid <= 0 or order_quote <= 0 or not depth:
        return float('inf')
    remaining = order_quote
    notional_spent = 0.0
    qty_filled = 0.0
    for price, amount in depth:
        p = float(price); a = float(amount)
        if p <= 0 or a <= 0:
            continue
        lvl_cap_q = p * a
        if lvl_cap_q >= remaining:
            qty = remaining / p
            qty_filled += qty
            notional_spent += qty * p
            remaining = 0.0
            break
        else:
            qty_filled += a
            notional_spent += lvl_cap_q
            remaining -= lvl_cap_q
    if qty_filled <= 0:
        return float('inf')
    vwap = notional_spent / qty_filled
    if side == 'buy':
        return (vwap / mid - 1.0) * 10000.0
    else:
        return (1.0 - vwap / mid) * 10000.0


def main():
    with open(os.path.join('configs', 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    scr = cfg['screener']

    ex = ccxt.coinbase({
        'enableRateLimit': True,
        'timeout': 15000,
        'options': {'adjustForTimeDifference': True},
        'apiKey': os.getenv('COINBASE_API_KEY') or None,
        'secret': os.getenv('COINBASE_API_SECRET') or None,
    })

    markets = ex.load_markets()
    sleep_rate_limit(ex)
    tickers = ex.fetch_tickers()

    # USD FX (cast for typing)
    fx = USDFX(scr.get('stablecoin_parity_warn_bps', 100.0))
    fx.load_from_tickers(cast(Dict[str, Dict[str, dict]], {'coinbase': tickers}))

    rows: List[Dict[str, Any]] = []
    reasons_count = Counter()

    # Build candidates with metrics
    for sym, t in tickers.items():
        # basic symbol check
        if '/' not in sym:
            continue
        base, quote = sym.split('/')
        # crude derivative filter
        if any(k in sym.upper() for k in ['PERP', 'SWAP', 'FUTURE', '-USD-']):
            continue
        # ensure it's a known spot market (if metadata exists)
        try:
            m = ex.market(sym)
            if m and not (m.get('spot') or (m.get('type') == 'spot')):
                continue
        except Exception:
            pass

        price = safe_float(t.get('last') or t.get('close'))
        bid = safe_float(t.get('bid'))
        ask = safe_float(t.get('ask'))

        # volume fallbacks
        qv = safe_float(t.get('quoteVolume'))
        if qv is None:
            bv = safe_float(t.get('baseVolume')) or safe_float(t.get('volume'))
            last = price
            if bv is not None and last is not None:
                qv = bv * last
        if qv is None:
            info = t.get('info') or {}
            # try common quote volume keys first
            for k in (
                'quoteVolume', 'quote_volume', 'volume_quote', 'volume_quote_24h',
                'quote_volume_24h', 'volume_usd', 'volume_usd_24h', 'usd_volume_24h'
            ):
                v = info.get(k)
                if v is not None:
                    try:
                        qv = float(v)
                        break
                    except Exception:
                        pass
            if qv is None:
                # base volume keys, convert to USD via last price
                for k in ('baseVolume','base_volume','base_volume_24h','volume','volume_24h'):
                    v = info.get(k)
                    if v is not None and price is not None:
                        try:
                            qv = float(v) * float(price)
                            break
                        except Exception:
                            pass
        usd_per_quote = fx.to_usd(quote) or get_usd_per_quote(ex, quote) or 1.0
        vol_usd = (qv * float(usd_per_quote)) if qv is not None else None

        spread_bps = None
        if bid is not None and ask is not None and bid > 0:
            spread_bps = ((ask - bid) / bid) * 10000.0

        # order book impact (best-effort)
        impact_bps = None
        try:
            ob = ex.fetch_order_book(sym, limit=10)
            bids_raw = ob.get('bids') or []
            asks_raw = ob.get('asks') or []
            bids: List[Tuple[float,float]] = []
            asks: List[Tuple[float,float]] = []
            for lvl in bids_raw[:10]:
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    p = safe_float(lvl[0]); a = safe_float(lvl[1])
                    if p is not None and a is not None:
                        bids.append((p, a))
            for lvl in asks_raw[:10]:
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    p = safe_float(lvl[0]); a = safe_float(lvl[1])
                    if p is not None and a is not None:
                        asks.append((p, a))
            if bids and asks:
                # derive spread from OB if missing
                if spread_bps is None and bids[0][0] > 0:
                    spread_bps = ((asks[0][0] - bids[0][0]) / bids[0][0]) * 10000.0
                mid = (bids[0][0] + asks[0][0]) / 2.0
                order_quote = float(scr['typical_order_usd']) / float(usd_per_quote or 1.0)
                buy_bps = side_impact_bps('buy', mid, asks, order_quote)
                sell_bps = side_impact_bps('sell', mid, bids, order_quote)
                impact_bps = max(buy_bps, sell_bps)
        except Exception:
            pass
        sleep_rate_limit(ex)

        # evaluate gates (record all reasons; do not drop rows)
        reasons = []
        if vol_usd is None:
            reasons.append('no_volume')
        elif vol_usd < float(scr['min_24h_volume_usd']):
            reasons.append('vol_usd<gate')
        if price is None or float(price) < float(scr['min_price']):
            reasons.append('price<min')
        if spread_bps is None:
            reasons.append('no_spread')
        elif spread_bps > float(scr['max_spread_bps']):
            reasons.append('spread>gate')
        if impact_bps is None:
            reasons.append('no_ob')
        elif impact_bps > float(scr['max_impact_bps']):
            reasons.append('impact>gate')
        if not reasons:
            reasons = ['PASS']
        for r in reasons:
            reasons_count[r] += 1

        rows.append({
            'symbol': sym,
            'base': base,
            'quote': quote,
            'price': float(price) if price is not None else None,
            'bid': float(bid) if bid is not None else None,
            'ask': float(ask) if ask is not None else None,
            'spread_bps': float(spread_bps) if spread_bps is not None else None,
            'quoteVolume': float(qv) if qv is not None else None,
            'usd_per_quote': float(usd_per_quote) if usd_per_quote is not None else None,
            'vol_usd': float(vol_usd) if vol_usd is not None else None,
            'impact_bps': float(impact_bps) if impact_bps is not None else None,
            'reasons': ';'.join(reasons)
        })

    # sort by vol_usd desc (None treated as 0) and take top 100
    rows.sort(key=lambda r: (r['vol_usd'] or 0.0), reverse=True)
    top = rows[:100]

    # write CSV
    out_dir = os.path.join('data', 'screener_snapshots', 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_path = os.path.join(out_dir, f'coinbase_top100_{stamp}.csv')
    fields = list(top[0].keys()) if top else ['symbol']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in top:
            w.writerow(r)

    # summary
    passed = sum(1 for r in top if r['reasons'] == 'PASS')
    print(f"Saved report: {out_path}")
    print(f"Top100 summary: PASS={passed} FAIL={len(top)-passed}")
    # reasons histogram (all markets)
    if reasons_count:
        print("Reason counts (all markets):")
        for k, v in reasons_count.most_common(10):
            print(f"  {k}: {v}")
    if top:
        print("Sample (top 10): symbol, vol_usd, spread_bps, impact_bps, reasons")
        for r in top[:10]:
            print(f"{r['symbol']}, {r['vol_usd']}, {r['spread_bps']}, {r['impact_bps']}, {r['reasons']}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(130)
    except Exception as e:
        print('Fatal:', e)
        sys.exit(1)
