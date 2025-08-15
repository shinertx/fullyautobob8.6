#!/bin/bash
set -euo pipefail
set -x # Enable debug output
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

# export .env if present
set -a
[ -f .env ] && source .env || true
set +a

# ensure Redis is accessible
echo "Checking for Redis..."
CONFIG_REDIS_PORT=$(grep 'redis_port' configs/config.yaml | awk '{print $2}')
if redis-cli -p $CONFIG_REDIS_PORT PING; then
    echo "Redis is responsive on port $CONFIG_REDIS_PORT."
else
    echo "Redis is not responsive on port $CONFIG_REDIS_PORT. Please ensure Redis is running."
    exit 1
fi

# venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# streamlit port from env or default 8601
: "${DASHBOARD_PORT:=8601}"

# launch in tmux
tmux has-session -t trading_session 2>/dev/null || tmux new-session -d -s trading_session "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; python -m v26meme.cli loop'"
tmux has-session -t dashboard_session 2>/dev/null || tmux new-session -d -s dashboard_session "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; streamlit run dashboard/app.py --server.fileWatcherType=none --server.port=${DASHBOARD_PORT}'"

# Hyper-Lab (Extreme Iteration Layer)
tmux has-session -t hyper_lab 2>/dev/null || tmux new-session -d -s hyper_lab "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; python -m v26meme.labs.hyper_lab run'"

echo "✅ v4.7.3 launched: tmux sessions [trading_session, dashboard_session, hyper_lab]"
echo " Attach: tmux attach -t trading_session | tmux attach -t dashboard_session | tmux attach -t hyper_lab"
echo " Detach: Ctrl+B then D"
