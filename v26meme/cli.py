import click, yaml, os, time, json, hashlib, random
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pandas as pd

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
from v26meme.data.universe_screener import UniverseScreener
from v26meme.data.screener_store import ScreenerStore
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator
from v26meme.research.validation import panel_cv_stats, benjamini_hochberg
from v26meme.research.feature_prober import FeatureProber
from v26meme.labs.simlab import SimLab
from v26meme.allocation.optimizer import PortfolioOptimizer
from v26meme.execution.exchange import ExchangeFactory
from v26meme.execution.handler import ExecutionHandler
from v26meme.execution.risk import RiskManager
from v26meme.core.dsl import Alpha
from v26meme.llm.proposer import LLMProposer
from v26meme.analytics.adaptive import publish_adaptive_knobs

def load_config(file="configs/config.yaml"):
    with open(file, "r") as f: return yaml.safe_load(f)

def lagged_zscore(s, lookback=200):
    m = s.rolling(lookback, min_periods=max(5, lookback//4)).mean().shift(1)
    sd = s.rolling(lookback, min_periods=max(5, lookback//4)).std().shift(1)
    return (s - m) / sd

def _ensure_lakehouse_bootstrap(cfg):
    tf = cfg['harvester_universe']['timeframe']
    d = Path("data") / tf
    have = list(d.glob("*.parquet"))
    days_override = os.getenv("BOOTSTRAP_HARVEST_DAYS")
    if days_override:
        cfg['harvester_universe']['initial_harvest_days'] = int(days_override)
    if not have:
        from data_harvester import harvest as _harvest
        import yaml as _yaml
        with open("configs/symbols.yaml", "r") as f:
            symmap = _yaml.safe_load(f)
        logger.info("No lakehouse detected; harvesting initial dataset (first run)...")
        _harvest(cfg, symmap)
        logger.info("Initial harvest complete.")

@click.group()
def cli(): pass

@cli.command()
def loop():
    load_dotenv()
    cfg = load_config()
    random.seed(cfg['system'].get('seed', 1337))
    Path("logs").mkdir(exist_ok=True, parents=True)
    logger.add("logs/system.log", level=cfg['system']['log_level'], rotation="10 MB", retention="14 days", enqueue=True)
    logger.info("🚀 v26meme v4.7.3 loop starting...")

    _ensure_lakehouse_bootstrap(cfg)

    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    lakehouse = Lakehouse()
    screener = UniverseScreener(
        exchanges=cfg['data_source']['exchanges'],
        min_24h_volume_usd=cfg['screener']['min_24h_volume_usd'],
        min_price=cfg['screener']['min_price'],
        max_markets=cfg['screener']['max_markets'],
        typical_order_usd=cfg['screener']['typical_order_usd'],
        max_spread_bps=cfg['screener']['max_spread_bps'],
        max_impact_bps=cfg['screener']['max_impact_bps'],
        impact_notional_usd=cfg['screener'].get('impact_notional_usd', 200)
    )
    # Inject adaptive/liquidity flags onto instance (backward compatible without changing ctor signature)
    screener.adaptive_gating_enabled = cfg['screener'].get('adaptive_gating_enabled', True)
    screener.liquidity_score_enabled = cfg['screener'].get('liquidity_score_enabled', True)
    screener.liquidity_score_percentile = cfg['screener'].get('liquidity_score_percentile', 0.25)
    store = ScreenerStore(cfg['screener'].get('snapshot_dir', 'data/screener_snapshots'))
    feature_factory = FeatureFactory()
    simlab = SimLab(cfg['execution']['paper_fees_bps'], cfg['execution']['paper_slippage_bps'])
    base_features = [
        'return_1p','volatility_20p','momentum_10p','rsi_14','close_vs_sma50',
        'hod_sin','hod_cos','round_proximity','btc_corr_20p','eth_btc_ratio'
    ]
    generator = GeneticGenerator(base_features, cfg['discovery']['population_size'], seed=cfg['system'].get('seed',1337))
    prober = FeatureProber(cfg['execution']['paper_fees_bps'], cfg['execution']['paper_slippage_bps'],
                           perturbations=cfg['prober'].get('perturbations',64),
                           delta_fraction=cfg['prober'].get('delta_fraction',0.15),
                           seed=cfg['system'].get('seed',1337))
    optimizer = PortfolioOptimizer(cfg)
    exchange_factory = ExchangeFactory(os.environ.get("GCP_PROJECT_ID"))
    risk = RiskManager(state, cfg)
    exec_handler = ExecutionHandler(state, exchange_factory, cfg, risk_manager=risk)
    # Initialize LLM proposer only if enabled and provider is not local
    proposer = None
    if cfg.get('llm', {}).get('enable', True) and cfg.get('llm', {}).get('provider', 'local') != 'local':
        proposer = LLMProposer(state)

    while True:
        try:
            state.heartbeat()
            logger.info("--- New loop cycle ---")
            instruments, tickers_by_venue = screener.get_active_universe()
            if not instruments:
                logger.warning("No instruments from screener; sleeping.")
                time.sleep(cfg['system']['loop_interval_seconds']); continue
            store.save(instruments, tickers_by_venue)

            # Adaptive knobs publish (uses BTC realized vol)
            tf = cfg['harvester_universe']['timeframe']
            btc_df = lakehouse.get_data("BTC_USD_SPOT", tf)
            publish_adaptive_knobs(state, cfg, btc_df)

            # Adjust discovery.population_size adaptively
            adapt_pop = state.get('adaptive:population_size')
            if adapt_pop:
                generator.population_size = int(adapt_pop)

            # map to lakehouse canonical keys (USD spot for research baseline)
            lh_syms = set(lakehouse.get_available_symbols(tf))
            tradeable = []
            for inst in instruments:
                canon = f"{inst['base']}_USD_SPOT"
                if canon in lh_syms:
                    tradeable.append((inst, canon))
            if not tradeable:
                logger.warning("No tradeable (intersection with lakehouse) this cycle; sleeping.")
                time.sleep(cfg['system']['loop_interval_seconds']); continue
            logger.info(f"Tradeable (research canonical): {[c for _,c in tradeable]}")

            # Discovery set construction
            if not generator.population: generator.initialize_population()

            # Pull EIL survivors
            if cfg.get('eil',{}).get('enabled', True):
                keys = [k for k in state.r.scan_iter(match="eil:candidates:*", count=100)]
                if keys:
                    for k in keys:
                        cand = state.get(k)
                        if cand and cand.get('formula'):
                            generator.population.append(cand['formula'])
                    # trim queue
                    for k in keys:
                        state.r.delete(k)

            # LLM sidecar suggestions (offline by default)
            if cfg.get('llm', {}).get('enable', True) and proposer is not None:
                k = int(cfg['llm'].get('max_suggestions_per_cycle', 3))
                for f in proposer.propose(base_features, k=k):
                    if f not in generator.population:
                        generator.population.append(f)

            # Preload BTC/ETH and panel features
            eth_df = lakehouse.get_data("ETH_USD_SPOT", tf)
            df_cache = {}
            bases = list({canon.split('_')[0] for _, canon in tradeable})
            panel_K = max(1, cfg['discovery']['panel_symbols'])
            chosen_bases = random.sample(bases, min(panel_K, len(bases)))
            for base in chosen_bases:
                canon = f"{base}_USD_SPOT"
                df = lakehouse.get_data(canon, tf)
                if df.empty: 
                    continue
                df_feat = feature_factory.create(df, symbol=canon, cfg=cfg, other_dfs={'BTC_USD_SPOT': btc_df, 'ETH_USD_SPOT': eth_df})
                for f in base_features:
                    df_feat[f] = lagged_zscore(df_feat[f], lookback=200)
                df_feat = df_feat.dropna()
                df_cache[canon] = df_feat

            # Evaluate genetic population quickly via panel CV stats
            fitness = {}
            for formula in generator.population:
                fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
                panel_returns = {}
                for base in chosen_bases:
                    canon = f"{base}_USD_SPOT"
                    dff = df_cache.get(canon)
                    if dff is None or dff.empty:
                        continue
                    stats = simlab.run_backtest(dff, formula)
                    overall = stats.get('all', {})
                    if overall and overall.get('n_trades',0)>0:
                        panel_returns[canon] = pd.Series(overall.get('returns', []), dtype=float)
                if not panel_returns:
                    fitness[fid] = 0.0
                    continue
                cv = panel_cv_stats(panel_returns, k_folds=cfg['discovery']['cv_folds'],
                                    embargo=cfg['discovery']['cv_embargo_bars'],
                                    alpha_fdr=cfg['discovery']['fdr_alpha'])
                fitness[fid] = float(cv['mean_oos'])

            # Promotion pass with BH-FDR on pseudo p-values derived from fitness
            rep_canon = next(iter(df_cache.keys()), None)
            promoted_candidates = []
            if rep_canon:
                df_rep = df_cache.get(rep_canon)
                prom = []
                for formula in generator.population:
                    fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
                    if df_rep is None: continue
                    stats_rep = simlab.run_backtest(df_rep, formula) if df_rep is not None else {}
                    overall = stats_rep.get('all', {})
                    if not overall or overall.get('n_trades',0)==0: 
                        continue
                    # monotone mapping fitness -> pseudo p-value
                    pval = 0.5 if fitness.get(fid, 0.0)<=0 else max(0.0, 1.0 - min(1.0, fitness[fid]*10))
                    prom.append({"fid": fid, "formula": formula, "p": pval, "overall": overall})
                if prom:
                    pvals = [c["p"] for c in prom]
                    kept_mask, _ = benjamini_hochberg(pvals, cfg['discovery']['fdr_alpha'])
                    promoted_candidates = [c for c, keep in zip(prom, kept_mask) if keep]

            # Hard gates + robustness prober + optional realism gate
            active_promos = []
            gate = cfg['discovery']['promotion_criteria']
            min_rob = float(cfg['prober'].get('min_robust_score', 0.55))
            for c in sorted(promoted_candidates, key=lambda x: x['overall']['sortino'], reverse=True):
                ov = c['overall']
                if ov.get('n_trades',0) >= gate['min_trades'] and \
                   ov.get('sortino',0) >= gate['min_sortino'] and \
                   ov.get('sharpe',0)  >= gate['min_sharpe']  and \
                   ov.get('win_rate',0)>= gate['min_win_rate'] and \
                   abs(ov.get('mdd',1.0)) <= gate['max_mdd']:
                    # Robustness prober
                    rob = prober.score(df_rep, c['formula'])
                    if rob.get("robust_score", 0.0) < min_rob:
                        continue
                    # attach execution instrument (match base)
                    base = rep_canon.split('_')[0]
                    chosen_inst = next((inst for inst,_ in tradeable if inst['base'] == base), tradeable[0][0])
                    alpha = Alpha(
                        id=c['fid'],
                        name=f"alpha_{c['fid'][:6]}",
                        formula=c['formula'],
                        universe=[f"{chosen_inst['base']}_USD_SPOT"],
                        instrument=chosen_inst,
                        timeframe=tf,
                        performance=simlab.run_backtest(df_rep, c['formula']) if df_rep is not None else {}
                    )
                    active_promos.append(alpha.dict())
                    # track gene usage/fitness
                    def _track(node):
                        if not isinstance(node[0], list):
                            selfeat, op, _thr = node
                            selfeat = str(selfeat).replace(" ","")
                            g = f"{selfeat}_{op}"
                            state.gene_incr(g, ov.get('sortino',0))
                        else:
                            _track(node[0]); _track(node[2])
                    _track(c['formula'])
                    if len(active_promos) >= cfg['discovery']['max_promotions_per_cycle']:
                        break

            # Evolve next generation
            generator.run_evolution_cycle(fitness)

            # Portfolio management
            active = state.get_active_alphas()
            seen = {a['id'] for a in active}
            for a in active_promos:
                if a['id'] not in seen: active.append(a)

            # Regime proxy: trend/vol on BTC
            regime = 'chop'
            if btc_df is not None and not btc_df.empty:
                rets = btc_df['close'].pct_change()
                if len(rets.dropna()) > 24:
                    regime = 'high_vol' if rets.rolling(24).std().iloc[-1] > rets.rolling(24).std().quantile(0.75) else 'chop'

            tw = optimizer.get_weights(active, regime)
            state.set("target_weights", tw)
            exec_handler.reconcile(tw, active)
            state.set_active_alphas(active)
            state.log_historical_equity(state.get_portfolio()['equity'])

            logger.info(f"Cycle done. Sleeping {cfg['system']['loop_interval_seconds']}s.")
            time.sleep(cfg['system']['loop_interval_seconds'])
        except KeyboardInterrupt:
            logger.warning("Shutdown requested.")
            break
        except Exception as e:
            logger.opt(exception=True).error(f"Loop error: {e}")
            if risk: risk.note_error()
            time.sleep(cfg['system']['loop_interval_seconds']*2)

if __name__ == '__main__':
    cli()
