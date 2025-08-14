README.md (v26meme v4.7.3 — Extreme Iteration Release)
Mission

Build a first‑principles, LLM‑driven, self‑improving crypto trading intelligence that compounds from a small base by discovering, validating, and scaling causal edges—autonomously, safely, and fast.

Prime Objective: Attempt $200 → $1,000,000 by maximizing compounding rate while obeying anti‑ruin risk rails.

Operating Mode: Paper-first (default), Live optional via config switch.

Philosophy: First principles > folklore. Causality > correlation. Breadth + reasoning > raw latency.

What’s New in 4.7.3 (vs 4.7.2)

Extreme Iteration Layer (EIL)

Micro‑experiments (dozens–hundreds/cycle) with budget caps and PIT snapshots.

Automated “hypothesis → experiment → score → archive” loop; every run produces high‑value meta‑data for alpha mining.

Market‑Mechanics Pack (causal features)

Order‑flow & liquidity proxies (impact, spread, depth‑lite from order book snapshots).

Microstructure regime tagging (trend/vol/chop + liquidity states).

Round‑number pressure & session‑time harmonics, PIT‑shifted.

Magic‑Number Purge & Adaptive Knobs

Replaced hardcoded thresholds with data‑driven baselines (rolling quantiles / stability‑tested).

Config equals guard‑rails; runtime selects values from recent distribution.

Causal Prober

Rule‑out tests (ticker/time rotation, label permutations, event ablation) to detect data‑leakage & spurious edges.

Micro‑Live Sandbox (Paper+)

Small, realistic mark‑to‑market with venue precision, min‑cost checks, and impact‑based sizing sanity mirroring real execution.

Safety/Risk Upgrades

Conserve mode now regime‑aware (volatility & liquidity aware scaling).

Kill‑switch tied to error bursts and anomalous slippage/impact spikes.

Lakehouse kept & improved

PIT‑safe parquet store; auto‑bootstrap on first run; harvester w/ retries & rate‑limit pacing.

System Architecture
Loop
 ├─ Screener (FX norm → volume/price/spread → impact gate → [optional] sentiment boost)
 ├─ Lakehouse (PIT data; 1h TF by default; auto-bootstrap)
 ├─ FeatureFactory (PIT-safe + market-mechanics pack; lagged z-scores)
 ├─ Generator (genetic) + LLM Proposer (local; remote optional)
 ├─ Extreme Iteration Layer (EIL): run micro-experiments, archive results
 ├─ Validation: Purged K-Fold CV + BH-FDR; Causal Prober (rule-out tests)
 ├─ Promotion Gates (min trades, Sortino, Sharpe, win rate, MDD, robustness)
 ├─ Portfolio Optimizer (inv-var + caps/floors) → Risk Manager (daily stop, floor, phases)
 └─ Execution Handler (paper/live; precision, costs, impact sanity)

Key Modules (high signal)

data/universe_screener.py — FX‐aware screener, spread & impact gates, optional sentiment boost, PIT snapshots.

data/lakehouse.py — PIT parquet store; data_harvester.py auto‑bootstrap on first run.

research/feature_factory.py — PIT‑clean core + market‑mechanics pack; lagged z‑scores.

research/generator.py — genetic formula search; seeded by llm/proposer.py (local offline; remote optional).

research/validation.py — Purged K‑Fold CV, BH‑FDR acceptance; panel OOS stats.

labs/simlab.py — PIT-safe backtests; regime bucketing; cost/slippage simulation.

labs/eil/* (part of 4.7.3) — Extreme Iteration Layer: experiments, scoring, archival.

labs/causal_prober.py — permutation & ablation tests to reject fake edges.

allocation/optimizer.py — inverse‑variance with concentration caps & floors.

execution/risk.py — daily stop, equity floor, symbol/gross caps, phases, conserve mode, kill‑switch.

execution/handler.py — Paper MTM + live path (market orders) with precision/min‑cost checks.

core/state.py — Redis state, equity curve, gene fitness/usage tracking.

dashboard/app.py — Equity, drawdown, allocations, actives, risk caps.

cli.py — Orchestrates the full loop (bootstrap → discovery → validation → promotion → allocation → exec).

Installation (fresh)
# On Ubuntu
sudo apt-get update -y
sudo apt-get install -y python3-venv redis-server tmux

# Clone your repo (or copy the source)
cd /home/user/v26meme
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


Set environment (create .env if missing):

COINBASE_API_KEY=""
COINBASE_API_SECRET=""
KRAKEN_API_KEY=""
KRAKEN_API_SECRET=""

CRYPTO_PANIC_TOKEN=""     # optional
ETHERSCAN_API_KEY=""      # optional
LLM_PROVIDER="local"      # or "openai"
OPENAI_API_KEY=""         # optional

BOOTSTRAP_HARVEST_DAYS=365
DASHBOARD_PORT=8601

First Run (paper mode)
# 1) Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# 2) Launch the trading loop & dashboard in tmux
tmux new -d -s trading_session "bash -lc 'cd /home/user/v26meme; source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; python -m v26meme.cli loop'"
tmux new -d -s dashboard_session "bash -lc 'cd /home/user/v26meme; source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; streamlit run dashboard/app.py --server.fileWatcherType=none --server.port=${DASHBOARD_PORT:-8601}'"


On first run, the loop auto‑harvests the Lakehouse if empty (PIT‑safe historical OHLCV).

Dashboard: http://<server-ip>:8601

Configuration (high‑leverage controls)

configs/config.yaml

screener: max_markets, max_spread_bps, max_impact_bps, sentiment_weight

discovery: population_size, panel_symbols, cv_folds, cv_embargo_bars, fdr_alpha, max_promotions_per_cycle, promotion_criteria{}

portfolio: max_alpha_concentration, min_allocation_weight

risk: daily_stop_pct, equity_floor_pct, max_symbol_weight, max_gross_weight, max_order_notional_usd, phases{}, conserve_mode{}

execution: mode: "paper" | "live", primary_exchange, paper_fees_bps, paper_slippage_bps

llm: provider, enable, max_suggestions_per_cycle

Operating Guide

Paper first: Let the system run until promotions appear and equity/drawdown curves stabilize.

Audit promotions: CV stats, BH‑FDR pass, gates all hit, causal prober not flagging.

Flip to live only after confidence: execution.mode: "live" and keep small caps; monitor risk rails & slippage.

Safety & Integrity

PIT Safety: All features are time‑shifted/lagged; no future leakage.

CV & FDR: Purged K‑Fold + BH-FDR reduce look‑ahead & multiple‑testing risk.

Kill‑switch: Error bursts OR anomalous slippage/impact spikes halt new exposure.

Conserve mode: Automatic exposure down‑scaling after adverse days/regimes.

Reproducibility: Pinned deps + deterministic seeds for research; execution remains real‑world.

Troubleshooting

No promotions: Ensure 1y of data harvested; increase panel_symbols; lightly relax min_trades (keep risk in mind).

Rate limits: Lower screener.max_markets; the screener already paces impacts with sleeps.

Redis issues: sudo systemctl start redis-server.

Dashboard port conflict: Set DASHBOARD_PORT in .env.

Roadmap (4.7.4+)

Multi‑TF PIT fusion (15m/1h/day) with proper temporal alignment.

Richer order‑book sampling (depth/imbalance) with rate‑limit budgets.

Perp/funding & borrow‑rate signals with carry decomposition.

DEX routing & on‑chain cost model (opt‑in).

License & Responsibility

This is research software. No guarantees of profits. Use paper mode to validate assumptions. If you go live, start tiny and accept full responsibility for outcomes.
