# Placeholder-safe, returns 0.0 unless configured with Etherscan. Kept minimal by design.
import os
class OnChainFlow:
    def __init__(self, min_notional_eth: float = 100.0):
        self.api_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
        self.min_notional = float(min_notional_eth)
    def whale_flow_rate(self, window_hours: int = 6) -> float:
        # Robust and safe: return 0.0 unless a richer pipeline is implemented.
        return 0.0
