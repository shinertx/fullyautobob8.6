import json, time
import redis
from typing import Any, Dict, List, Tuple
from loguru import logger

class StateManager:
    def __init__(self, host='localhost', port=6379):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        try:
            self.r.ping()
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to Redis: {e}")
            raise

    def set(self, key: str, value: Any): self.r.set(key, json.dumps(value))
    def get(self, key: str):
        val = self.r.get(key)
        return json.loads(val) if val else None

    def heartbeat(self): self.r.set('heartbeat_ts', int(time.time()))

    def get_portfolio(self) -> Dict[str, Any]:
        return self.get('portfolio') or {'cash': 200.0, 'equity': 200.0, 'positions': {}}
    def set_portfolio(self, portfolio: Dict[str, Any]): self.set('portfolio', portfolio)

    def log_historical_equity(self, equity: float):
        ts = int(time.time())
        self.r.zadd('equity_curve', {json.dumps({'ts': ts, 'equity': equity}): ts})

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        return [json.loads(v) for v in self.r.zrange('equity_curve', 0, -1)]

    def get_active_alphas(self) -> List[Dict[str, Any]]: return self.get('active_alphas') or []
    def set_active_alphas(self, alphas: List[Dict[str, Any]]): self.set('active_alphas', alphas)

    # gene tracking
    def gene_incr(self, gene: str, fitness: float):
        self.r.zincrby('gene_usage', 1, gene)
        self.r.hincrbyfloat('gene_fitness', gene, float(fitness))
    def gene_top(self, min_count=10, top_n=20) -> List[Tuple[str, float]]:
        genes = self.r.hgetall('gene_fitness')
        out = []
        for g, totfit in genes.items():
            cnt = self.r.zscore('gene_usage', g) or 0
            if cnt >= min_count:
                out.append((g, float(totfit)/float(cnt)))
        return sorted(out, key=lambda x: x[1], reverse=True)[:top_n]
