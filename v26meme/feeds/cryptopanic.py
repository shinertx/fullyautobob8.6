import os, time, requests
from typing import Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API = "https://cryptopanic.com/api/v1/posts/"
AN = SentimentIntensityAnalyzer()

def _score_text(t): return AN.polarity_scores(t or "").get("compound", 0.0)

class CryptoPanicFeed:
    def __init__(self, window_hours=6, min_score=-1.0):
        self.token = os.environ.get("CRYPTO_PANIC_TOKEN", "").strip()
        self.window_hours = window_hours
        self.min_score = min_score

    def scores_by_ticker(self, tickers: List[str]) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {t: [] for t in tickers}
        if not self.token or not tickers:
            return out
        now = int(time.time())
        params = {"auth_token": self.token, "public": "true", "currencies": ",".join(tickers), "kind": "news", "page": 1}
        try:
            r = requests.get(API, params=params, timeout=10); r.raise_for_status()
            data = r.json().get("results", [])
        except Exception:
            return out
        cutoff = now - int(self.window_hours * 3600)
        for item in data:
            ts = item.get("published_at")
            if not ts: continue
            import time as _time
            ts = int(_time.mktime(_time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")))
            if ts < cutoff: continue
            title = item.get("title", "")
            sc = _score_text(title)
            if sc < self.min_score: continue
            cur = item.get("currencies") or []
            for c in cur:
                sym = (c.get("code") or "").upper()
                if sym in out:
                    out[sym].append({"ts": ts, "score": float(sc)})
        return out
