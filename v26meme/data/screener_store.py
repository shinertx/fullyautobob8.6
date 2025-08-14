import json, time
from pathlib import Path
from typing import List, Dict, Any

def _compact_ticker(t: dict) -> dict:
    keys = ['symbol','timestamp','datetime','bid','ask','last','close','quoteVolume','baseVolume']
    return {k: t.get(k) for k in keys if k in t}

class ScreenerStore:
    def __init__(self, snapshot_dir: str = "data/screener_snapshots"):
        self.dir = Path(snapshot_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, instrument_list: List[Dict[str, Any]], tickers_by_venue: Dict[str, Dict[str, dict]]):
        ts = int(time.time())
        fp = self.dir / f"{ts}.json"
        compact = {v: {s: _compact_ticker(t) for s, t in d.items()} for v, d in tickers_by_venue.items()}
        payload = {"ts": ts, "universe": instrument_list, "tickers": compact}
        fp.write_text(json.dumps(payload))
        return ts, str(fp)

    def load_all(self):
        items = []
        for p in sorted(self.dir.glob("*.json")):
            try: items.append(json.loads(p.read_text()))
            except Exception: pass
        return items
