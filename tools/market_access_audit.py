#!/usr/bin/env python3
"""
Market-Access Audit for v26meme 4.7.3 (paper-safe, read-only)

Validates connectivity, symbol mapping, precision/limits, ticker/OB sanity, FX parity, and rate-limit hygiene
for configured venues & canonical symbols.

No orders are placed. Calls are paced using each venue's rateLimit.
"""
from __future__ import annotations

import os
import sys
import time
import math
import json
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

# Third-party
import yaml
import ccxt  # type: ignore

# --- Helpers -----------------------------------------------------------------

def utc_ms() -> int:
    return int(time.time() * 1000)

def ms_to_iso(ms: Optional[int]) -> str:
    if ms is None:
        return "None"
    try:
        return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat()
    except Exception:
        return str(ms)


def sleep_rate_limit(ex, factor: float = 1.15):
    rl = getattr(ex, 'rateLimit', 1000) or 1000
    time.sleep(max(0.001, rl/1000.0 * factor))


def find_pair(ex, base_quote: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (resolved_symbol, note) trying aliases (e.g., Kraken XBT/USD for BTC/USD)."""
    if base_quote in ex.markets:
        return base_quote, None
    # Try a few well-known aliases
    base, quote = base_quote.split('/')
    candidates = []
    if base.upper() == 'BTC':
        candidates.append('XBT/' + quote)
    if base.upper() == 'XBT':
        candidates.append('BTC/' + quote)
    for cand in candidates:
        if cand in ex.markets:
            return cand, f"Alias used: {cand} for {base_quote}"
    return None, "Not listed"


def get_usd_per_quote(ex, quote: Optional[str]) -> Optional[float]:
    """Get USD per 1 unit of quote currency using venue tickers (USDT/USDC/DAI supported)."""
    if not quote or quote.upper() == 'USD':
        return 1.0
    q = quote.upper()
    pair = f"{q}/USD"
    inv_pair = f"USD/{q}"
    try:
        if pair in ex.markets:
            t = ex.fetch_ticker(pair)
            sleep_rate_limit(ex)
            return float(t.get('last') or 1.0)
        if inv_pair in ex.markets:
            t = ex.fetch_ticker(inv_pair)
            sleep_rate_limit(ex)
            r = t.get('last')
            return float(1.0/ r) if r else None
    except Exception:
        return None
    return None


def side_impact_bps(side: str, mid: float, depth: List[Dict[str, float]], order_quote: float) -> float:
    """Compute impact in bps against mid for spending order_quote units of quote currency on one side of book.
    depth: list of [price, amount] as dicts with keys 'price', 'amount'
    """
    if mid <= 0 or order_quote <= 0 or not depth:
        return float('inf')
    remaining = order_quote
    notional_spent = 0.0  # in quote currency
    qty_filled = 0.0
    for lvl in depth:
        p = float(lvl['price'])
        a = float(lvl['amount'])
        if p <= 0 or a <= 0:
            continue
        lvl_cap_quote = p * a  # quote currency capacity at this level
        if lvl_cap_quote >= remaining:
            qty = remaining / p
            qty_filled += qty
            notional_spent += qty * p
            remaining = 0.0
            break
        else:
            qty_filled += a
            notional_spent += lvl_cap_quote
            remaining -= lvl_cap_quote
    if qty_filled <= 0:
        return float('inf')
    avg_price = notional_spent / qty_filled  # in quote currency
    if side == 'buy':
        return (avg_price / mid - 1.0) * 10000.0
    else:
        return (1.0 - avg_price / mid) * 10000.0


def ob_monotonic_sane(bids: List[List[float]], asks: List[List[float]]) -> Tuple[bool, str]:
    # Monotonicity
    ok = True
    msg = []
    # bids: non-increasing prices
    for i in range(1, min(10, len(bids))):
        if bids[i][0] > bids[i-1][0]:
            ok = False; msg.append(f"bids[{i}].price > bids[{i-1}].price")
            break
    # asks: non-decreasing prices
    for i in range(1, min(10, len(asks))):
        if asks[i][0] < asks[i-1][0]:
            ok = False; msg.append(f"asks[{i}].price < asks[{i-1}].price")
            break
    # top-5 depth > 0 (avoid tuple unpacking; Kraken returns extra columns)
    bid_top5 = sum((lvl[1] if len(lvl) > 1 else 0) for lvl in bids[:5]) if bids else 0
    ask_top5 = sum((lvl[1] if len(lvl) > 1 else 0) for lvl in asks[:5]) if asks else 0
    if bid_top5 <= 0 or ask_top5 <= 0:
        ok = False; msg.append("top-5 depth empty")
    return ok, "; ".join(msg)


@dataclass
class CheckRow:
    check: str
    venue_symbol: str
    status: str
    evidence: str
    fix: str = ""


@dataclass
class VenueSummary:
    venue: str
    markets_count: int
    rate_limit_ms: int
    skew_ms: Optional[int]


def main():
    # Load configs
    with open(os.path.join('configs', 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join('configs', 'symbols.yaml'), 'r') as f:
        symmap = yaml.safe_load(f)

    venues: List[str] = cfg['data_source']['exchanges']
    canons: List[str] = cfg['harvester_universe']['symbols']
    screener = cfg['screener']

    # Build per-venue exchange clients
    ex_objs: Dict[str, Any] = {}
    for v in venues:
        cls = getattr(ccxt, v)
        ex = cls({
            'enableRateLimit': True,
            'timeout': 10000,
            'options': {
                'adjustForTimeDifference': True,
            }
        })
        ex_objs[v] = ex

    qa_rows: List[CheckRow] = []
    venue_summaries: List[VenueSummary] = []
    ticker_samples: Dict[str, List[Dict[str, Any]]] = {}
    fx_parity: Dict[str, List[Dict[str, Any]]] = {}
    latency_errors: Dict[str, Dict[str, Any]] = {v: {'latencies_ms': [], '429s': 0} for v in venues}

    # Connectivity & Time Skew
    for v, ex in ex_objs.items():
        try:
            t0 = utc_ms(); markets = ex.load_markets(); t1 = utc_ms()
            markets_count = len(markets)
            rl = getattr(ex, 'rateLimit', 1000) or 1000
            # Skew via fetch_time if available
            skew_ms = None
            try:
                sleep_rate_limit(ex)
                server_ms = ex.fetch_time()
                if isinstance(server_ms, (int, float)):
                    skew_ms = int(abs(server_ms - utc_ms()))
            except Exception:
                skew_ms = None
            venue_summaries.append(VenueSummary(venue=v, markets_count=markets_count, rate_limit_ms=rl, skew_ms=skew_ms))
            status = 'PASS' if markets_count>0 and (skew_ms is None or skew_ms <= 2000) else 'FAIL'
            qa_rows.append(CheckRow('Connectivity & Skew', v, status, f"markets={markets_count}, rateLimit={rl}ms, skew={skew_ms}"))
        except Exception as e:
            qa_rows.append(CheckRow('Connectivity & Skew', v, 'FAIL', f"error={type(e).__name__}: {e}", fix="Check network/ccxt"))
        finally:
            sleep_rate_limit(ex)

    # Symbol Mapping Conformance and Ticker/OB checks
    for canon in canons:
        base = canon.split('_')[0]
        for v, ex in ex_objs.items():
            vkey = f"{v}:{canon}"
            # Resolve mapping
            mapping = symmap.get(canon, {})
            ex_symbol = mapping.get(v)
            note = None
            # If explicitly unmapped/null -> treat as documented not-listed
            if ex_symbol in (None, '', 'not listed'):
                qa_rows.append(CheckRow('Symbol Mapping', vkey, 'PASS', 'configured not listed', fix=''))
                continue
            try:
                ex.load_markets()
            except Exception:
                pass
            if not ex_symbol:
                # try a reasonable default
                ex_symbol = f"{base}/USD"
            if ex_symbol not in ex.markets:
                resolved, note = find_pair(ex, ex_symbol)
                ex_symbol = resolved
            # Fallback to stable-quote if USD not listed
            if not ex_symbol:
                for q in ['USDT','DAI','USDC']:
                    cand = f"{base}/{q}"
                    if cand in ex.markets:
                        ex_symbol = cand
                        note = f"Quote fallback used: {cand}"
                        break
            if not ex_symbol:
                qa_rows.append(CheckRow('Symbol Mapping', vkey, 'FAIL', 'Not listed', fix='Update symbols.yaml or remove'))
                continue
            m = ex.market(ex_symbol)
            # Type/spot
            is_spot = bool(m.get('spot') or (m.get('type')=='spot'))
            prec = m.get('precision') or {}
            lims = m.get('limits') or {}
            has_prec = ('price' in prec) and ('amount' in prec)
            has_min = (lims.get('cost', {}) or {}).get('min') is not None or (lims.get('amount', {}) or {}).get('min') is not None
            status = 'PASS' if is_spot and has_prec and has_min else 'FAIL'
            fix = '' if status=='PASS' else 'Ensure spot market with precision & min limits; adjust mapping'
            ev = f"spot={is_spot}, precision={list(prec.keys())}, min_cost={lims.get('cost',{}).get('min')}, min_amt={lims.get('amount',{}).get('min')}"
            if note:
                ev += f" | note: {note}"
            qa_rows.append(CheckRow('Symbol Mapping', vkey, status, ev, fix))
            if status=='FAIL':
                continue

            # Ticker Integrity: two samples
            try:
                t1 = utc_ms(); tkr1 = ex.fetch_ticker(ex_symbol); t2 = utc_ms()
                sleep_rate_limit(ex)
                t3 = utc_ms(); tkr2 = ex.fetch_ticker(ex_symbol); t4 = utc_ms()
                lat1 = t2 - t1; lat2 = t4 - t3
                latency_errors[v]['latencies_ms'].extend([lat1, lat2])
                # sanity
                bid1, ask1, last1, ts1 = tkr1.get('bid'), tkr1.get('ask'), tkr1.get('last'), tkr1.get('timestamp')
                bid2, ask2, last2, ts2 = tkr2.get('bid'), tkr2.get('ask'), tkr2.get('last'), tkr2.get('timestamp')
                age1 = abs(utc_ms() - (ts1 or 0)) if ts1 else None
                age2 = abs(utc_ms() - (ts2 or 0)) if ts2 else None
                ok = True; issues = []
                if not (bid1 and ask1 and bid1 < ask1 and (last1 or 0) > 0): ok=False; issues.append('tick1 invalid')
                if not (bid2 and ask2 and bid2 < ask2 and (last2 or 0) > 0): ok=False; issues.append('tick2 invalid')
                if age1 is not None and age1 > 60000: ok=False; issues.append(f"age1={age1}ms")
                if age2 is not None and age2 > 60000: ok=False; issues.append(f"age2={age2}ms")
                if last1 and last2 and last1>0:
                    delta = abs(last2-last1)/last1
                    if delta > 0.10: ok=False; issues.append(f"|Δlast|={delta:.1%}")
                status = 'PASS' if ok else 'FAIL'
                ev = f"{ex_symbol} t1: bid/ask/last=({bid1},{ask1},{last1}) age={age1}ms; t2: ({bid2},{ask2},{last2}) age={age2}ms; lat={lat1}/{lat2}ms"
                qa_rows.append(CheckRow('Ticker Integrity', vkey, status, ev, fix='Investigate data freshness' if status=='FAIL' else ''))
                ticker_samples.setdefault(vkey, []).append({
                    'symbol': ex_symbol, 't1': {'ts': ts1, 'dt': ms_to_iso(ts1), 'bid': bid1, 'ask': ask1, 'last': last1},
                    't2': {'ts': ts2, 'dt': ms_to_iso(ts2), 'bid': bid2, 'ask': ask2, 'last': last2}, 'lat_ms': [lat1, lat2]
                })
            except ccxt.RateLimitExceeded:
                latency_errors[v]['429s'] += 1
                qa_rows.append(CheckRow('Ticker Integrity', vkey, 'FAIL', 'RateLimitExceeded', fix='Increase spacing vs rateLimit'))
            except Exception as e:
                qa_rows.append(CheckRow('Ticker Integrity', vkey, 'FAIL', f"error={type(e).__name__}: {e}", fix='Check mapping/venue status'))
            finally:
                sleep_rate_limit(ex)

            # Order Book Sanity & Impact
            try:
                ob = ex.fetch_order_book(ex_symbol, limit=10)
                bids = ob.get('bids') or []
                asks = ob.get('asks') or []
                ok, msg = ob_monotonic_sane(bids, asks)
                bid = bids[0][0] if bids else None
                ask = asks[0][0] if asks else None
                mid = ((bid or 0) + (ask or 0))/2 if bid and ask else None
                impact_ok = False
                impact_ev = ''
                if mid:
                    # prepare structured depth (avoid tuple unpacking)
                    bids_s = []
                    for lvl in bids[:10]:
                        if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                            try:
                                bids_s.append({'price': float(lvl[0]), 'amount': float(lvl[1])})
                            except Exception:
                                continue
                    asks_s = []
                    for lvl in asks[:10]:
                        if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                            try:
                                asks_s.append({'price': float(lvl[0]), 'amount': float(lvl[1])})
                            except Exception:
                                continue
                    # compute USD-per-quote and convert typical_order_usd to quote units
                    try:
                        m2 = ex.market(ex_symbol)
                    except Exception:
                        m2 = {}
                    quote_ccy = (m2.get('quote') if isinstance(m2, dict) else None) or (ex_symbol.split('/')[-1] if '/' in ex_symbol else 'USD')
                    usd_per_quote = get_usd_per_quote(ex, quote_ccy) or 1.0
                    order_quote = float(screener['typical_order_usd']) / float(usd_per_quote)
                    buy_bps = side_impact_bps('buy', mid, asks_s, order_quote)
                    sell_bps = side_impact_bps('sell', mid, bids_s, order_quote)
                    worst = max(buy_bps, sell_bps)
                    impact_ok = worst <= screener['max_impact_bps']
                    impact_ev = f"impact_bps buy/sell={buy_bps:.1f}/{sell_bps:.1f} worst={worst:.1f} (quote={quote_ccy}, usd_per_quote={usd_per_quote:.4f})"
                status = 'PASS' if ok and impact_ok else 'FAIL'
                ev = f"best bid/ask=({bid},{ask}); sanity={ok} {msg or ''}; {impact_ev}"
                qa_rows.append(CheckRow('OrderBook & Impact', vkey, status, ev, fix='Exclude or raise impact gate' if status=='FAIL' else ''))
            except ccxt.RateLimitExceeded:
                latency_errors[v]['429s'] += 1
                qa_rows.append(CheckRow('OrderBook & Impact', vkey, 'FAIL', 'RateLimitExceeded', fix='Increase spacing vs rateLimit'))
            except Exception as e:
                qa_rows.append(CheckRow('OrderBook & Impact', vkey, 'FAIL', f"error={type(e).__name__}: {e}", fix='Check symbol availability'))
            finally:
                sleep_rate_limit(ex)

    # FX Parity (stablecoins vs USD)
    stables = ['USDT','USDC','DAI']
    for v, ex in ex_objs.items():
        fx_parity[v] = []
        for st in stables:
            pair = f"{st}/USD"
            inv_pair = f"USD/{st}"
            rate = None; used = None
            try:
                if pair in ex.markets:
                    t = ex.fetch_ticker(pair); rate = t.get('last'); used = pair
                elif inv_pair in ex.markets:
                    t = ex.fetch_ticker(inv_pair); r = t.get('last'); rate = 1.0/r if r else None; used = inv_pair
                else:
                    used = 'not listed'
                if rate:
                    off_bps = abs(rate - 1.0) * 10000.0
                else:
                    off_bps = None
                fx_parity[v].append({'stable': st, 'pair': used, 'rate': rate, 'off_bps': off_bps})
            except Exception as e:
                fx_parity[v].append({'stable': st, 'pair': used or 'error', 'rate': None, 'off_bps': None, 'error': str(e)})
            finally:
                sleep_rate_limit(ex)

    # Min-Cost & Precision Round-Trip (primary_exchange)
    primary = cfg['execution']['primary_exchange']
    primary_ex = ex_objs.get(primary)
    mincost_rows: List[CheckRow] = []
    if primary_ex:
        for canon in canons[:3]:  # check BTC/ETH/SOL minimum trio
            base = canon.split('_')[0]
            mapping = symmap.get(canon, {})
            ex_symbol = mapping.get(primary) or f"{base}/USD"
            if ex_symbol not in primary_ex.markets:
                ex_symbol, _ = find_pair(primary_ex, ex_symbol)
            if not ex_symbol:
                mincost_rows.append(CheckRow('Min-Cost & Precision', f"{primary}:{canon}", 'FAIL', 'symbol missing', fix='Update symbols.yaml'))
                continue
            m = primary_ex.market(ex_symbol)
            prec = m.get('precision') or {}
            lims = m.get('limits') or {}
            price = None
            try:
                t = primary_ex.fetch_ticker(ex_symbol); price = t.get('last')
            except Exception:
                pass
            sleep_rate_limit(primary_ex)
            min_cost = (lims.get('cost', {}) or {}).get('min')
            if min_cost is None and price:
                amin = (lims.get('amount', {}) or {}).get('min')
                min_cost = price * amin if amin else None
            ok = bool(price and min_cost and cfg['screener']['typical_order_usd'] >= min_cost)
            # round trip using ccxt helpers (precision-aware)
            try:
                ex_price = primary_ex.price_to_precision(ex_symbol, price) if price else None
            except Exception:
                ex_price = price
            try:
                amt_guess = (cfg['screener']['typical_order_usd']/(price or 1.0)) if price else None
                ex_amt = primary_ex.amount_to_precision(ex_symbol, amt_guess) if amt_guess else None
            except Exception:
                ex_amt = amt_guess if price else None
            ev = f"{ex_symbol} min_cost={min_cost}; price={ex_price} amt={ex_amt}"
            mincost_rows.append(CheckRow('Min-Cost & Precision', f"{primary}:{canon}", 'PASS' if ok else 'FAIL', ev, fix='Raise typical_order_usd or adjust pair' if not ok else ''))

    # Throughput & 429 Hygiene: small repeated pulls on BTC/ETH
    for v, ex in ex_objs.items():
        for ex_symbol in ['BTC/USD','ETH/USD']:
            if ex_symbol not in ex.markets:
                ex_symbol, _ = find_pair(ex, ex_symbol)
            if not ex_symbol:
                continue
            for _ in range(3):
                t0 = utc_ms()
                try:
                    ex.fetch_ticker(ex_symbol)
                except ccxt.RateLimitExceeded:
                    latency_errors[v]['429s'] += 1
                except Exception:
                    pass
                t1 = utc_ms()
                latency_errors[v]['latencies_ms'].append(t1-t0)
                sleep_rate_limit(ex)

    # Output
    print("\n=== Venue Summary ===")
    for vs in venue_summaries:
        print(f"- {vs.venue}: markets={vs.markets_count}, rateLimit={vs.rate_limit_ms}ms, skew_ms={vs.skew_ms}")

    print("\n=== QA Matrix ===")
    print("Check | Venue/Symbol | PASS/FAIL | Evidence | Recommended Fix")
    for row in qa_rows + mincost_rows:
        print(f"{row.check} | {row.venue_symbol} | {row.status} | {row.evidence} | {row.fix}")

    print("\n=== FX Parity ===")
    warn_bps = cfg['screener']['stablecoin_parity_warn_bps']
    for v, rows in fx_parity.items():
        print(f"- {v}:")
        for r in rows:
            flag = ''
            if r.get('off_bps') is not None and r['off_bps'] > warn_bps:
                flag = ' (WARN)'
            print(f"  {r['stable']}: pair={r['pair']} rate={r['rate']} off_bps={r['off_bps']}{flag}")

    print("\n=== Latency & Errors ===")
    for v, stats in latency_errors.items():
        lat = stats['latencies_ms']
        if lat:
            p50 = sorted(lat)[len(lat)//2]
            print(f"- {v}: pulls={len(lat)} p50={p50}ms 429s={stats['429s']}")
        else:
            print(f"- {v}: no samples 429s={stats['429s']}")

    # Minimal pass/fail gates to proceed to system audit
    # 1) All mapped canons resolve or are documented
    mapping_fail = any(row.check=='Symbol Mapping' and row.status=='FAIL' for row in qa_rows)
    # 2) Ticker/OB gates met on BTC/ETH/SOL per venue
    def pass_for(symbol: str, check: str) -> bool:
        return any((row.venue_symbol.endswith(symbol) and row.check==check and row.status=='PASS') for row in qa_rows)
    trio_ok = True
    for s in ['BTC_USD_SPOT','ETH_USD_SPOT','SOL_USD_SPOT']:
        for v in venues:
            key = f"{v}:{s}"
            t_ok = any(row.venue_symbol==key and row.check=='Ticker Integrity' and row.status=='PASS' for row in qa_rows)
            ob_ok = any(row.venue_symbol==key and row.check=='OrderBook & Impact' and row.status=='PASS' for row in qa_rows)
            trio_ok = trio_ok and t_ok and ob_ok
    # 3) FX parity within bounds or pair absent
    parity_ok = True
    for v, rows in fx_parity.items():
        for r in rows:
            if r['pair'] == 'not listed':
                continue
            if r.get('off_bps') is None:
                continue
            if r['off_bps'] > warn_bps:
                parity_ok = False
    overall = 'PASS' if (not mapping_fail and trio_ok and parity_ok) else 'FAIL'
    print(f"\n=== OVERALL: {overall} ===")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
        sys.exit(1)
