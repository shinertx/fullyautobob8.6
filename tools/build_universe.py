#!/usr/bin/env python3
"""
Auto-build expanded canonical universe for Coinbase & Kraken (paper-safe).

Logic (updated):
- Load screener config + adaptive/liquidity switches.
- For each exchange: load markets, fetch all tickers once.
- Pre-filter spot USD / stable (USDT/USDC/DAI) quotes by absolute floors: price, volume, spread.
- Fetch order books ONLY for pre-filtered candidates to derive depth5 + impact (typical_order_usd).
- Compute composite liquidity score (vol, depth, inverse spread, inverse impact).
- Apply absolute safety floors again, then adaptive percentile (liq_score >= quantile) if enabled.
- Dedupe by base (choose highest score per base) and build canonical BASE_USD_SPOT mapping.
- Persist list to generated_universe.yaml and merged symbol mappings to symbols.generated.yaml.

PIT Safety: Point-in-time snapshot only; no look-ahead.
"""
from __future__ import annotations
import os, math, yaml, time, sys
from typing import Dict, Any, List, Tuple
import ccxt  # type: ignore
import pandas as pd
from v26meme.data.liquidity_score import get_depth5_usd, get_spread_bps, get_impact_bps, calculate_liquidity_scores

GEN_UNIVERSE_PATH = "configs/generated_universe.yaml"
GEN_SYMBOLS_PATH  = "configs/symbols.generated.yaml"
STABLES = ["USDT","USDC","DAI"]

# --- Helpers ---

def load_cfg() -> Dict[str, Any]:
    with open("configs/config.yaml","r") as f:
        return yaml.safe_load(f)

def sleep_rl(ex, factor=1.0):
    rl = getattr(ex, 'rateLimit', 1000) or 1000
    time.sleep(max(0.0005, (rl/1000.0) * factor))

# --- Main ---

def main():
    cfg = load_cfg()
    screener = cfg.get("screener", {})
    min_vol = float(screener.get("min_24h_volume_usd", 5_000_000))  # Absolute volume floor
    min_price = float(screener.get("min_price", 0.2))               # Absolute price floor
    max_spread = float(screener.get("max_spread_bps", 25))          # Absolute spread ceiling
    max_impact = float(screener.get("max_impact_bps", 60))          # Absolute impact ceiling
    typical_order_usd = float(screener.get("typical_order_usd", 200))
    adaptive_gating = bool(screener.get("adaptive_gating_enabled", True))
    liquidity_score_enabled = bool(screener.get("liquidity_score_enabled", True))
    liq_score_pct = float(screener.get("liquidity_score_percentile", 0.25))
    max_markets = int(screener.get("max_markets", 240))

    venues: List[str] = cfg.get("data_source", {}).get("exchanges", ["coinbase","kraken"])  # type: ignore

    print("Collecting liquidity metrics across all markets...")
    sys.stdout.flush()

    # 1. Load exchanges + tickers
    ex_objs: Dict[str, Any] = {}
    tickers_by_venue: Dict[str, Dict[str, Any]] = {}
    for v in venues:
        cls = getattr(ccxt, v)
        ex = cls({'enableRateLimit': True, 'timeout': 15000})
        ex.load_markets()
        sleep_rl(ex)
        try:
            tickers = ex.fetch_tickers()
        except Exception as e:
            print(f"WARN: fetch_tickers failed on {v}: {e}")
            tickers = {}
        ex_objs[v] = ex
        tickers_by_venue[v] = tickers

    # 2. Pre-filter inexpensive metrics (price, volume, spread from tickers)
    prelim: List[Dict[str, Any]] = []
    for v, tickers in tickers_by_venue.items():
        for sym, t in tickers.items():
            if '/' not in sym: continue
            base, quote = sym.split('/')[:2]
            if quote.upper() not in ["USD"] + STABLES: continue
            mdef = ex_objs[v].markets.get(sym, {})
            if not (mdef.get('spot') or mdef.get('type') == 'spot'): continue
            last = t.get('last') or t.get('close') or t.get('bid') or t.get('ask')
            if not last or last <= 0: continue
            if last < min_price: continue
            bid = t.get('bid'); ask = t.get('ask')
            if not bid or not ask or bid <= 0 or ask <= 0: continue
            sp_bps = ((ask - bid)/bid) * 10000.0
            if sp_bps > max_spread: continue
            # Volume extraction
            info = t.get('info') or {}
            vol_candidates = [
                ('quoteVolume', t.get('quoteVolume')),
                ('quote_volume', info.get('quote_volume')),
                ('volume_usd', info.get('volume_usd')),
                ('volume_usd_24h', info.get('volume_usd_24h')),
            ]
            vol_usd = None
            for k, val in vol_candidates:
                if val is not None:
                    try:
                        vol_usd = float(val)
                        break
                    except:  # noqa
                        pass
            # Fallback: baseVolume * last
            if vol_usd is None:
                base_vol = t.get('baseVolume') or info.get('baseVolume') or info.get('base_volume')
                if base_vol and last:
                    try:
                        vol_usd = float(base_vol) * float(last)
                    except:  # noqa
                        pass
            if not vol_usd or vol_usd < min_vol:  # absolute floor
                continue
            prelim.append({
                'venue': v,
                'symbol': sym,
                'base': base.upper(),
                'quote': quote.upper(),
                'price': float(last),
                'vol_24h_usd': float(vol_usd),
                'spread_bps': float(sp_bps),
            })
    if not prelim:
        print("No markets passed preliminary absolute floors.")
        return

    # 3. Fetch order books only for prelim list to derive depth + impact
    metrics: List[Dict[str, Any]] = []
    for i, row in enumerate(prelim):
        ex = ex_objs[row['venue']]
        ob = None
        try:
            ob = ex.fetch_order_book(row['symbol'], limit=10)
        except Exception:
            pass
        depth5 = get_depth5_usd(ob) if ob else 0.0
        impact_bps = get_impact_bps(ob, typical_order_usd) if ob else float('inf')
        # If impact or depth missing we still keep; gating later will drop
        metrics.append({
            **row,
            'depth5_usd': depth5,
            'impact_bps': impact_bps
        })
        if (i+1) % 25 == 0:
            print(f"  OB progress: {i+1}/{len(prelim)}")
            sys.stdout.flush()
        sleep_rl(ex, factor=0.6)

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        print("No metrics collected.")
        return

    # 4. Composite scoring & gating
    if liquidity_score_enabled:
        scored = calculate_liquidity_scores(metrics_df)
        # Absolute safety floors
        gated = scored[(scored['vol_24h_usd'] >= min_vol) &
                       (scored['spread_bps'] <= max_spread) &
                       (scored['impact_bps'] <= max_impact) &
                       (scored['depth5_usd'] > 0)]
        if adaptive_gating and not gated.empty:
            threshold = gated['liq_score'].quantile(liq_score_pct)
            gated = gated[gated['liq_score'] >= threshold]
        # Deduplicate by base selecting highest liq_score
        gated = gated.sort_values('liq_score', ascending=False)
    else:
        # Legacy gating fallback (no composite score)
        gated = metrics_df[(metrics_df['vol_24h_usd'] >= min_vol) &
                           (metrics_df['spread_bps'] <= max_spread) &
                           (metrics_df['impact_bps'] <= max_impact) &
                           (metrics_df['depth5_usd'] > 0)]
        gated = gated.sort_values('vol_24h_usd', ascending=False)

    selected_rows: List[Dict[str, Any]] = []
    seen_bases = set()
    for _, r in gated.iterrows():
        b = r['base']
        if b in seen_bases:  # keep first (already highest score/order)
            continue
        seen_bases.add(b)
        selected_rows.append(r.to_dict())
        if len(selected_rows) >= max_markets:
            break

    if not selected_rows:
        print("All candidates filtered out after composite gating.")
        return

    # 5. Build canonical mapping
    expanded: Dict[str, Dict[str, str]] = {}
    for r in selected_rows:
        canon = f"{r['base']}_USD_SPOT"
        expanded.setdefault(canon, {})[r['venue']] = r['symbol']

    # 6. Merge with existing manual symbols
    symbols_yaml_path = "configs/symbols.yaml"
    try:
        with open(symbols_yaml_path, "r") as f:
            existing = yaml.safe_load(f) or {}
    except FileNotFoundError:
        existing = {}
    merged = dict(existing)
    for k, v in expanded.items():
        merged.setdefault(k, v)

    # 7. Persist outputs
    generated_symbols = sorted(expanded.keys())
    with open(GEN_UNIVERSE_PATH, "w") as f:
        yaml.safe_dump({"symbols": generated_symbols}, f, sort_keys=True)
    with open(GEN_SYMBOLS_PATH, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=True)

    print(f"Generated canons: {len(generated_symbols)} (new only)")
    print(f"Wrote {GEN_UNIVERSE_PATH} and {GEN_SYMBOLS_PATH}.")
    print("Sample (first 20):", generated_symbols[:20])

if __name__ == "__main__":
    # Allow user mistake of PNPATH / YTHONPATH by just warning
    if 'PNPATH' in os.environ or 'YTHONPATH' in os.environ:
        print("NOTE: Use PYTHONPATH=. python3 tools/build_universe.py (detected possible typo).")
    main()
