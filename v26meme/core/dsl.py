from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class Alpha(BaseModel):
    id: str
    name: str
    formula: List[Any]                 # boolean expression on features
    universe: List[str]                # canonical lakehouse symbols (e.g., BTC_USD_SPOT)
    instrument: Optional[Dict[str, Any]] = None  # execution instrument metadata
    timeframe: str
    performance: Dict[str, Dict[str, Any]] = {}
    model_config = ConfigDict(frozen=True)
