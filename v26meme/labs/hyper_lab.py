import time, json, random, hashlib
import click, yaml
import pandas as pd
from loguru import logger
from pathlib import Path
import os

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator
from v26meme.labs.simlab import SimLab

def load_config(file="configs/config.yaml"):
    with open(file, "r") as f: return yaml.safe_load(f)

def _parse_tf_bars(tf: str, days: int) -> int:
    # rough: '1h' => 24*days bars; '15m' => 96*days bars
    if tf.endswith('h'):
        return int(tf[:-1]) and (24//int(tf[:-1]))*days if int(tf[:-1])>0 else 24*days
    if tf.endswith('m'):
        m = int(tf[:-1]); per_day = 24*60//max(1,m)
        return per_day*days
    return 24*days

@click.group()
def cli(): pass

@cli.command()
@click.option("--once", is_flag=True, help="Run a single EIL cycle and exit.")
@click.pass_context
def run(ctx, once: bool = False):
    cfg = load_config()
    if not cfg.get('eil',{}).get('enabled', True):
        logger.info("EIL disabled; exiting.")
        return

    random.seed(cfg['system'].get('seed', 1337))
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    lake = Lakehouse()
    ff = FeatureFactory()
    sim = SimLab(cfg['execution']['paper_fees_bps'], cfg['execution']['paper_slippage_bps'])

    base_features = [
        'return_1p','volatility_20p','momentum_10p','rsi_14','close_vs_sma50',
        'hod_sin','hod_cos','round_proximity','btc_corr_20p','eth_btc_ratio'
    ]
    gen = GeneticGenerator(base_features, population_size=cfg['discovery']['population_size'], seed=cfg['system'].get('seed', 1337))

    tf = cfg['harvester_universe']['timeframe']
    fast_days = cfg['eil']['fast_window_days']
    nbars = _parse_tf_bars(tf, fast_days)
    cycle_seconds = int(os.environ.get("EIL_CYCLE_SECONDS", "30"))
    tested = 0

    start_time = time.time()
    while True:
        try:
            # sample panel
            avail = lake.get_available_symbols(tf)
            if not avail:
                time.sleep(10); continue
            bases = [s for s in avail if s.endswith("_USD_SPOT")]
            if not bases:
                time.sleep(10); continue
            panel = random.sample(bases, min(cfg['discovery']['panel_symbols'], len(bases)))

            # load features
            df_cache = {}
            btc = lake.get_data("BTC_USD_SPOT", tf)
            eth = lake.get_data("ETH_USD_SPOT", tf)
            for canon in panel:
                df = lake.get_data(canon, tf)
                if df.empty: continue
                df = df.tail(nbars)
                df_feat = ff.create(df, symbol=canon, cfg=cfg, other_dfs={'BTC_USD_SPOT': btc, 'ETH_USD_SPOT': eth})
                df_cache[canon] = df_feat.dropna()

            if not gen.population: gen.initialize_population()

            tested = len(gen.population)
            # evaluate quickly
            survivors = []
            for f in gen.population:
                fid = hashlib.sha256(json.dumps(f).encode()).hexdigest()
                perfs = []
                for canon, dff in df_cache.items():
                    stats = sim.run_backtest(dff, f).get('all', {})
                    if stats and stats.get('n_trades',0)>0:
                        perfs.append(stats.get('avg_return', 0.0))
                if not perfs:
                    continue
                score = sum(perfs)/max(1,len(perfs))
                survivors.append((score, fid, f))

            survivors.sort(key=lambda x: x[0], reverse=True)
            topk = survivors[: cfg['eil']['survivor_top_k']]
            # publish to Redis
            for score, fid, form in topk:
                state.set(f"eil:candidates:{fid}", {"fid": fid, "formula": form, "score": score, "ts": int(time.time())})
                state.r.sadd('eil:seen_hashes', fid)
            # telemetry
            state.set("eil:metrics", {"tested": tested, "survivors": len(topk), "ts": int(time.time())})

            # evolve
            fitness = {fid: score for score, fid, _ in survivors}
            gen.run_evolution_cycle(fitness)

            if once:
                logger.info("EIL --once flag set, exiting after one cycle.")
                break

            elapsed = time.time() - start_time
            if elapsed < cycle_seconds:
                time.sleep(cycle_seconds - elapsed)
            start_time = time.time()

        except KeyboardInterrupt:
            logger.warning("EIL shutdown requested.")
            break
        except Exception as e:
            logger.opt(exception=True).error(f"EIL loop error: {e}")
            time.sleep(cycle_seconds)


if __name__ == "__main__":
    cli()
