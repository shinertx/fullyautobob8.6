from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class AssetID:
    source: str
    symbol: str
    chain_id: Optional[int] = None
    address: Optional[str] = None
    decimals: Optional[int] = None

@dataclass(frozen=True)
class InstrumentID:
    venue: str
    type: str  # "spot" | "swap" | "future"
    market_id: str
    base: AssetID
    quote: AssetID
    precision: Dict[str, Any]
    limits: Dict[str, Any]
    display: str

def make_instrument(venue: str, market: dict) -> 'InstrumentID':
    typ = "spot" if not market.get("swap") and not market.get("future") else ("swap" if market.get("swap") else "future")
    base_sym = market.get("base", "UNKNOWN")
    quote_sym = market.get("quote", "UNKNOWN")
    display = f"{base_sym}_{quote_sym}_{'SPOT' if typ=='spot' else typ.upper()}"
    return InstrumentID(
        venue=venue,
        type=typ,
        market_id=market.get("symbol", ""),
        base=AssetID(source="ccxt", symbol=base_sym),
        quote=AssetID(source="ccxt", symbol=quote_sym),
        precision=market.get("precision", {}),
        limits=market.get("limits", {}),
        display=display
    )
