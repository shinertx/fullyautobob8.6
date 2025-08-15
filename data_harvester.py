import yaml, time, pandas as pd, ccxt
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, List, Any

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

_ex_cache: Dict[str, Any] = {}

def get_exchange(exchange_id):
    if exchange_id in _ex_cache:
        return _ex_cache[exchange_id]
    try:
        ex = getattr(ccxt, exchange_id)({'enableRateLimit': True, 'timeout': 15000})
        ex.load_markets()
        _ex_cache[exchange_id] = ex
        return ex
    except Exception as e:
        logger.error(f"Failed to init {exchange_id}: {e}")
        return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _fetch_ohlcv(ex, symbol, tf, since, limit):
    return ex.fetch_ohlcv(symbol, tf, since, limit=limit)

def load_symbol_map() -> Dict[str, Dict[str,str]]:
    base = {}
    gen = {}
    try:
        with open('configs/symbols.yaml','r') as f:
            base = yaml.safe_load(f) or {}
    except Exception:
        logger.warning("symbols.yaml missing or invalid")
    try:
        with open('configs/symbols.generated.yaml','r') as f:
            gen = yaml.safe_load(f) or {}
    except Exception:
        pass
    merged = dict(base)
    for k,v in gen.items():
        if k not in merged:
            merged[k] = v
        else:
            merged[k].update({kk:vv for kk,vv in v.items() if kk not in merged[k]})
    return merged

def harvest(cfg, symbol_map=None):
    logger.info("Starting Data Lakehouse harvester (multi-venue)...")
    if symbol_map is None:
        symbol_map = load_symbol_map()
    lake_path = Path("./data")
    tf = cfg['harvester_universe']['timeframe']
    days = cfg['harvester_universe']['initial_harvest_days']
    exchanges: List[str] = cfg['data_source']['exchanges']
    primary = cfg['execution'].get('primary_exchange', exchanges[0])
    mode = cfg.get('harvester_universe', {}).get('multi_venue_mode', 'fallback').lower()
    min_history_days = cfg['harvester_universe'].get('min_history_days', 0)
    max_gap_factor = cfg['harvester_universe'].get('max_gap_factor', 1.5)
    qc_dir = Path(cfg['harvester_universe'].get('qc_output_dir', 'data/qc'))
    per_venue_sidecar = cfg['harvester_universe'].get('per_venue_sidecar', False)
    qc_dir.mkdir(parents=True, exist_ok=True)
    primary_ex = get_exchange(primary)
    if primary_ex is None:
        logger.error("Primary exchange unavailable; aborting harvest.")
        return
    tf_ms = primary_ex.parse_timeframe(tf) * 1000
    expected_delta = tf_ms
    qc_rows = []

    for symbol in cfg['harvester_universe']['symbols']:
        mappings = symbol_map.get(symbol, {}) or {}
        available = [(ex_id, mappings.get(ex_id)) for ex_id in exchanges if mappings.get(ex_id)]
        if not available:
            logger.warning(f"No venue mapping for {symbol}")
            continue
        if mode not in ('fallback','aggregate'):
            mode = 'fallback'

        if mode == 'fallback':
            # pick primary if present else first
            chosen_ex_id, ex_sym = None, None
            for ex_id, ex_symbol in available:
                if ex_id == primary:
                    chosen_ex_id, ex_sym = ex_id, ex_symbol; break
            if chosen_ex_id is None:
                chosen_ex_id, ex_sym = available[0]
            ex = get_exchange(chosen_ex_id)
            if ex is None:
                logger.error(f"Exchange init failed for {chosen_ex_id}; skipping {symbol}")
                continue
            logger.info(f"Harvesting {symbol} ({ex_sym}) on {chosen_ex_id} (mode=fallback)...")
            since = ex.milliseconds() - 86400000 * days
            last = None; rows = []
            try:
                while since < ex.milliseconds():
                    ohlcv = _fetch_ohlcv(ex, ex_sym, tf, since, limit=1000)
                    if not ohlcv: break
                    if last is not None and ohlcv[-1][0] <= last: break
                    last = ohlcv[-1][0]
                    rows.extend(ohlcv)
                    since = last + tf_ms
                    time.sleep(ex.rateLimit/1000.0)
            except Exception as e:
                logger.error(f"Fetch failed for {symbol} on {chosen_ex_id}: {e}")
                continue
            if not rows:
                logger.warning(f"No data for {symbol} on {chosen_ex_id}")
                continue
            df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
        else:  # aggregate
            parts = []
            for ex_id, ex_sym in available:
                ex = get_exchange(ex_id)
                if ex is None: continue
                logger.info(f"Harvest slice {symbol} ({ex_sym}) from {ex_id} (mode=aggregate)...")
                since = ex.milliseconds() - 86400000 * days
                last=None; rows=[]
                try:
                    while since < ex.milliseconds():
                        ohlcv = _fetch_ohlcv(ex, ex_sym, tf, since, limit=1000)
                        if not ohlcv: break
                        if last is not None and ohlcv[-1][0] <= last: break
                        last = ohlcv[-1][0]
                        rows.extend(ohlcv)
                        since = last + tf_ms
                        time.sleep(ex.rateLimit/1000.0)
                except Exception as e:
                    logger.error(f"Fetch failed for {symbol} on {ex_id}: {e}")
                    continue
                if not rows: continue
                dfi = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
                dfi['exchange'] = ex_id
                parts.append(dfi)
            if not parts:
                logger.warning(f"No data for {symbol} across venues")
                continue
            # Align & aggregate
            merged = None
            for dfi in parts:
                dfi['timestamp'] = pd.to_datetime(dfi['timestamp'], unit='ms', utc=True)
                dfi = dfi.set_index('timestamp')
                if merged is None:
                    merged = dfi
                else:
                    merged = merged.join(dfi, how='outer', rsuffix=f"_{dfi['exchange'].iloc[0]}")
            if merged is None or merged.empty:
                logger.warning(f"No merged data for {symbol}")
                continue
            volume_cols = [c for c in merged.columns if c.startswith('volume')]
            open_cols = [c for c in merged.columns if c.startswith('open')]
            high_cols = [c for c in merged.columns if c.startswith('high')]
            low_cols  = [c for c in merged.columns if c.startswith('low')]
            close_cols= [c for c in merged.columns if c.startswith('close')]
            def first_valid(row, cols):
                for c in cols:
                    val = row.get(c)
                    if pd.notna(val): return val
                return None
            out = pd.DataFrame(index=merged.index)
            out['open']  = merged[open_cols].apply(lambda r: first_valid(r, open_cols), axis=1)
            out['high']  = merged[high_cols].max(axis=1)
            out['low']   = merged[low_cols].min(axis=1)
            out['close'] = merged[close_cols].apply(lambda r: r.dropna().iloc[-1] if r.dropna().size>0 else None, axis=1)
            out['volume']= merged[volume_cols].sum(axis=1)
            df = out.reset_index()
        # Common post-processing
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) if df['timestamp'].dtype!='datetime64[ns, UTC]' else df['timestamp']
        df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
        # Quality guard: drop rows with any NaN OHLC
        df = df.dropna(subset=['open','high','low','close'])
        # QC: minimum history length
        span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 if len(df) else 0
        if span_days < min_history_days:
            logger.warning(f"QC reject {symbol}: only {span_days:.1f} days < min_history_days={min_history_days}")
            continue
        # QC: gap detection
        gaps = df.index.to_series().diff().dt.total_seconds().fillna(expected_delta/1000.0)
        large_gaps = gaps[gaps > (max_gap_factor * expected_delta/1000.0)]
        gap_ratio = (large_gaps.count() / max(1, len(df)))
        qc_rows.append({
            'symbol': symbol,
            'rows': len(df),
            'days_span': span_days,
            'large_gap_count': int(large_gaps.count()),
            'large_gap_ratio': gap_ratio
        })
        if gap_ratio > 0.02:  # 2% tolerance
            logger.warning(f"QC note {symbol}: large_gap_ratio={gap_ratio:.2%}")
        # Persist canonical
        outdir = lake_path / tf
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outdir / f"{symbol}.parquet")
        # Optional per-venue sidecar for fallback mode (only single ex used)
        if per_venue_sidecar and mode == 'fallback':
            side_dir = lake_path / tf / 'per_venue'
            side_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(side_dir / f"{symbol}__{chosen_ex_id}.parquet")
        logger.success(f"Saved {len(df)} rows for {symbol} ({mode}) span={span_days:.1f}d gaps={gap_ratio:.2%}")
    # Write QC report
    if qc_rows:
        import json
        (qc_dir / 'harvest_qc.json').write_text(json.dumps(qc_rows, indent=2))
        logger.info(f"QC report written: {qc_dir / 'harvest_qc.json'}")

if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    harvest(cfg)
