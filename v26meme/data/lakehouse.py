from pathlib import Path
import pandas as pd

class Lakehouse:
    def __init__(self, data_dir: str = "./data"):
        self.base_path = Path(data_dir)

    def load_instrument(self, canonical_symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Alias for get_data to provide a more intuitive method name."""
        return self.get_data(canonical_symbol, timeframe)

    def get_data(self, canonical_symbol: str, timeframe: str) -> pd.DataFrame:
        fp = self.base_path / timeframe / f"{canonical_symbol}.parquet"
        return pd.read_parquet(fp) if fp.exists() else pd.DataFrame()

    def get_available_symbols(self, timeframe: str) -> list:
        p = self.base_path / timeframe
        if not p.exists(): return []
        return [q.stem for q in p.glob("*.parquet")]
