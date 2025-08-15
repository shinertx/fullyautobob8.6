import os
import time
import loguru
import ccxt
from v26meme.data.usd_fx import USDFX

class UniverseScreener:
    def __init__(self, 
                 exchanges=['coinbase', 'kraken'],
                 min_24h_volume_usd=10_000_000,
                 min_price=0.20,
                 max_markets=24,
                 typical_order_usd=200,
                 max_spread_bps=25,
                 max_impact_bps=60,
                 impact_notional_usd=200):
        # Normalize in case a full screener dict was mistakenly passed into any numeric arg
        def _extract(val, key, fallback):
            if isinstance(val, dict):
                loguru.logger.warning(f"UniverseScreener: '{key}' received a dict; extracting '{key}' from it. Check caller config passing.")
                return val.get(key, fallback)
            return val

        self.exchange_names = exchanges  # Store names, not objects
        self.min_24h_volume_usd = _extract(min_24h_volume_usd, 'min_24h_volume_usd', 10_000_000)
        self.min_price = _extract(min_price, 'min_price', 0.20)
        self.max_markets = int(_extract(max_markets, 'max_markets', 24))
        self.typical_order_usd = _extract(typical_order_usd, 'typical_order_usd', 200)
        self.max_spread_bps = _extract(max_spread_bps, 'max_spread_bps', 25)
        self.max_impact_bps = _extract(max_impact_bps, 'max_impact_bps', 60)
        self.impact_notional_usd = _extract(impact_notional_usd, 'impact_notional_usd', 200)
        
        # Debug logging to catch the issue
        loguru.logger.debug(f"UniverseScreener init: min_24h_volume_usd = {self.min_24h_volume_usd} (type: {type(self.min_24h_volume_usd)})")
        
        # Ensure numeric values
        if not isinstance(self.min_24h_volume_usd, (int, float)):
            raise ValueError(f"min_24h_volume_usd must be numeric, got {type(self.min_24h_volume_usd)}: {self.min_24h_volume_usd}")

    def _impact_bps(self, ex, market_id, notional_usd, usd_per_quote):
        """Calculate price impact in basis points for a given notional trade."""
        try:
            ob = ex.fetch_order_book(market_id, limit=10)
        except Exception as e:
            loguru.logger.debug(f"Could not fetch order book for {market_id}: {e}")
            return float('inf')
        
        asks = ob.get('asks', [])
        bids = ob.get('bids', [])
        
        if not asks or not bids:
            return float('inf')
        
        # Extract price and quantity, handling variable-length tuples
        mid = (bids[0][0] + asks[0][0]) / 2.0
        filled = 0
        total_cost = 0
        notional_base = notional_usd / usd_per_quote
        
        for price, qty, *_ in asks:  # Use *_ to handle extra elements
            can_fill = min(qty, notional_base - filled)
            filled += can_fill
            total_cost += can_fill * price
            if filled >= notional_base:
                break
        
        if filled < notional_base * 0.95:
            return float('inf')
        
        vwap = total_cost / filled if filled > 0 else mid
        impact = abs((vwap - mid) / mid) * 10000
        return impact

    def get_active_universe(self, lakehouse=None) -> tuple[list, dict]:
        """Returns (instruments, tickers_by_venue)"""
        import pandas as pd
        from v26meme.data.liquidity_score import get_depth5_usd, get_spread_bps, get_impact_bps, calculate_liquidity_scores
        logger = loguru.logger.bind(module="universe_screener")
        all_candidates = []
        tickers_by_venue = {}
        # Load config switches
        adaptive_gating = getattr(self, 'adaptive_gating_enabled', True)
        liquidity_score_enabled = getattr(self, 'liquidity_score_enabled', True)
        liquidity_score_percentile = getattr(self, 'liquidity_score_percentile', 0.25)
        # Hysteresis support: recall last threshold if available
        last_thr = None
        try:
            if os.path.exists('data/screener_snapshots/last_liq_threshold.txt'):
                with open('data/screener_snapshots/last_liq_threshold.txt','r') as f:
                    last_thr = float(f.read().strip())
        except Exception:
            last_thr = None
        # Process each exchange
        for exchange_name in self.exchange_names:
            logger.info(f"Screening {exchange_name}...")
            try:
                exchange_class = getattr(ccxt, exchange_name)
                ex = exchange_class({
                    "apiKey": os.getenv(f"{exchange_name.upper()}_API_KEY"),
                    "secret": os.getenv(f"{exchange_name.upper()}_API_SECRET")
                })
                if hasattr(ex, 'rateLimit'):
                    time.sleep(ex.rateLimit / 1000.0)
                all_tickers = ex.fetch_tickers()
                tickers_by_venue[exchange_name] = all_tickers
            except Exception as e:
                logger.warning(f"Could not fetch tickers from {exchange_name}: {e}")
                continue
        fx = USDFX()
        fx.load_from_tickers(tickers_by_venue)
        # Collect candidate metrics
        for exchange_name, all_tickers in tickers_by_venue.items():
            try:
                exchange_class = getattr(ccxt, exchange_name)
                ex = exchange_class({
                    "apiKey": os.getenv(f"{exchange_name.upper()}_API_KEY"),
                    "secret": os.getenv(f"{exchange_name.upper()}_API_SECRET")
                })
            except Exception as e:
                logger.warning(f"Could not create exchange {exchange_name}: {e}")
                continue
            for symbol, ticker in all_tickers.items():
                if not ticker.get('symbol') or '/' not in ticker['symbol']:
                    continue
                parts = ticker['symbol'].split('/')
                if len(parts) != 2:
                    continue
                base, quote = parts
                if any(kw in symbol.upper() for kw in ['PERP', 'SWAP', 'FUTURE', '-USD-']):
                    continue
                price = ticker.get('last')
                if price is None or price <= 0:
                    continue
                quote_vol = ticker.get('quoteVolume')
                if quote_vol is None and ticker.get('baseVolume') and price:
                    quote_vol = ticker['baseVolume'] * price
                if quote_vol is None or quote_vol <= 0:
                    continue
                try:
                    usd_rate = fx.to_usd(quote)
                    vol_usd = quote_vol * usd_rate if usd_rate else quote_vol
                except:
                    vol_usd = quote_vol
                ob = None
                try:
                    ob = ex.fetch_order_book(symbol, limit=10)
                except Exception:
                    pass
                depth5_usd = get_depth5_usd(ob) if ob else 0.0
                spread_bps_val = get_spread_bps(ob) if ob else float('inf')
                impact_bps_val = get_impact_bps(ob, self.typical_order_usd) if ob else float('inf')
                all_candidates.append({
                    'ts': int(time.time()),
                    'venue': exchange_name,
                    'symbol': symbol,
                    'base': base,
                    'quote': quote,
                    'price': price,
                    'vol_24h_usd': vol_usd,
                    'depth5_usd': depth5_usd,
                    'spread_bps': spread_bps_val,
                    'impact_bps': impact_bps_val
                })
        metrics_df = pd.DataFrame(all_candidates)
        # Persist raw snapshot for audit
        try:
            if not metrics_df.empty:
                snap_dir = 'data/screener_snapshots'
                os.makedirs(snap_dir, exist_ok=True)
                metrics_df.to_parquet(f"{snap_dir}/metrics_{int(time.time())}.parquet")
        except Exception as e:
            logger.debug(f"Could not persist metrics snapshot: {e}")
        selected = []
        if liquidity_score_enabled and not metrics_df.empty:
            scored_df = calculate_liquidity_scores(metrics_df)
            filtered = scored_df[
                (scored_df['vol_24h_usd'] >= self.min_24h_volume_usd) &
                (scored_df['depth5_usd'] >= self.min_24h_volume_usd) &
                (scored_df['spread_bps'] <= self.max_spread_bps) &
                (scored_df['impact_bps'] <= self.max_impact_bps)
            ]
            if adaptive_gating and not filtered.empty:
                thr = filtered['liq_score'].quantile(liquidity_score_percentile)
                # Simple hysteresis: if last_thr exists and new thr within 5% relative, blend
                if last_thr is not None and thr is not None and last_thr > 0:
                    if abs(thr - last_thr)/last_thr < 0.05:
                        thr = 0.5*thr + 0.5*last_thr
                filtered = filtered[filtered['liq_score'] >= thr]
                try:
                    with open('data/screener_snapshots/last_liq_threshold.txt','w') as f:
                        f.write(f"{thr}")
                except Exception:
                    pass
            filtered = filtered.sort_values('liq_score', ascending=False)
            seen_bases = set()
            for _, row in filtered.iterrows():
                if row['base'] in seen_bases:
                    continue
                seen_bases.add(row['base'])
                inst = {
                    'venue': row['venue'],
                    'market_id': row['market_id'],
                    'base': row['base'],
                    'quote': row['quote'],
                    'price': row['price'],
                    'volume_usd': row['vol_24h_usd'],
                    'spread_bps': row['spread_bps'],
                    'impact_bps': row['impact_bps'],
                    'liq_score': row.get('liq_score')
                }
                selected.append(inst)
                if len(selected) >= self.max_markets:
                    break
        else:
            # Fallback to legacy gating
            if not metrics_df.empty:
                metrics_df = metrics_df.sort_values('vol_24h_usd', ascending=False)
                seen_bases = set()
                for _, row in metrics_df.iterrows():
                    if row['base'] in seen_bases:
                        continue
                    seen_bases.add(row['base'])
                    if (
                        row['vol_24h_usd'] >= self.min_24h_volume_usd and
                        row['depth5_usd'] >= self.min_24h_volume_usd and
                        row['spread_bps'] <= self.max_spread_bps and
                        row['impact_bps'] <= self.max_impact_bps
                    ):
                        inst = {
                            'venue': row['venue'],
                            'market_id': row['market_id'],
                            'base': row['base'],
                            'quote': row['quote'],
                            'price': row['price'],
                            'volume_usd': row['vol_24h_usd'],
                            'spread_bps': row['spread_bps'],
                            'impact_bps': row['impact_bps']
                        }
                        selected.append(inst)
                        if len(selected) >= self.max_markets:
                            break
        if not selected:
            logger.warning("Liquidity/impact screening removed all markets.")
            # Paper mode fallback: create synthetic instruments from common symbols when no live data available
            paper_symbols = ['BTC_USD_SPOT', 'ETH_USD_SPOT', 'SOL_USD_SPOT', 'ADA_USD_SPOT', 'AVAX_USD_SPOT']
            logger.info("Falling back to paper mode instruments for development/testing")
            for symbol in paper_symbols[:min(5, self.max_markets)]:
                base = symbol.split('_')[0]
                selected.append({
                    'venue': 'synthetic',
                    'market_id': symbol,
                    'base': base,
                    'quote': 'USD',
                    'price': 1.0,  # Will be updated from historical data
                    'volume_usd': 10_000_000,  # Synthetic volume
                    'spread_bps': 10,  # Synthetic spread
                    'impact_bps': 30,  # Synthetic impact
                    'liq_score': 0.8  # Synthetic liquidity score
                })
        else:
            logger.info(f"Active universe: {[inst['base'] for inst in selected]}")
        return selected, tickers_by_venue
