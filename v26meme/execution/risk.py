from loguru import logger
import time

class RiskManager:
    def __init__(self, state, cfg):
        self.state, self.cfg = state, cfg['risk']
        self.adapt_cfg = cfg.get('adaptive', {}) or {}
        self._ensure_day_anchor()

    def _ensure_day_anchor(self):
        ref = self.state.get('risk:day_anchor')
        now = int(time.time())
        if not ref or now - ref.get('ts',0) > 86400:
            eq = self.state.get_portfolio().get('equity', 200.0)
            self.state.set('risk:day_anchor', {'ts': now, 'equity': eq})
            self.state.set('risk:halted', False)
            self.state.set('risk:errors', 0)

    def note_error(self):
        n = (self.state.get('risk:errors') or 0) + 1
        self.state.set('risk:errors', n)
        if n >= self.cfg['max_consecutive_errors']:
            self.state.set('risk:halted', True)
            logger.error(f"Kill-switch engaged after {n} consecutive errors.")

    def reset_errors(self):
        self.state.set('risk:errors', 0)

    def _daily_stop_dynamic(self, equity: float) -> float:
        ds = float(self.cfg.get('daily_stop_pct', 0.10))
        if self.adapt_cfg.get('enabled', True):
            dyn = self.state.get('adaptive:daily_stop_pct')
            if dyn is not None:
                floor = float(self.adapt_cfg.get('daily_stop_pct_floor', 0.05))
                ceil = float(self.adapt_cfg.get('daily_stop_pct_ceiling', 0.20))
                ds = max(floor, min(ceil, float(dyn)))
        return ds

    def enforce(self, symbol_weights: dict, equity: float) -> dict:
        self._ensure_day_anchor()

        start_equity = self.state.get('risk:initial_equity') or equity
        if not self.state.get('risk:initial_equity'):
            self.state.set('risk:initial_equity', start_equity)
        if equity < start_equity * (1.0 - self.cfg['equity_floor_pct']):
            self.state.set('risk:halted', True)
            logger.error("Equity floor breached. Halting trading.")
            return {}

        day_anchor = self.state.get('risk:day_anchor') or {'equity': equity}
        dd = (equity - day_anchor['equity']) / max(1e-9, day_anchor['equity'])
        daily_stop = self._daily_stop_dynamic(equity)
        if dd <= -daily_stop:
            self.state.set('risk:halted', True)
            logger.warning(f"Daily stop hit ({dd:.2%}). Halting trading.")
            return {}

        # Conserve mode on severe drawdown
        cm = self.cfg.get('conserve_mode', {})
        gross_scalar = 1.0
        kelly_scalar = 1.0
        if dd <= -float(cm.get('dd_trigger_pct', 0.15)):
            gross_scalar = float(cm.get('gross_weight_scalar', 0.5))
            kelly_scalar = float(cm.get('kelly_scalar', 0.6))

        if self.state.get('risk:halted'):
            logger.warning("Risk halted. Skipping new exposure.")
            return {}

        # Phase overrides (live scaling)
        current_kf = self.cfg.get('kelly_fraction', 0.5) * kelly_scalar
        current_max_order = self.cfg.get('max_order_notional_usd', 150)
        phases = self.cfg.get('phases', {}) or {}
        for thresh in sorted([float(k) for k in phases.keys()]):
            if equity >= float(thresh):
                current_kf = phases[str(int(thresh))].get('kelly_fraction', current_kf)
                current_max_order = phases[str(int(thresh))].get('max_order_notional_usd', current_max_order)
        # publish current caps (dashboard reads)
        self.state.set('risk:current_max_order', current_max_order)
        self.state.set('risk:current_kelly_fraction', current_kf)

        # enforce symbol caps & gross scalar
        capped = {}
        total = 0.0
        for sym, w in symbol_weights.items():
            w = max(0.0, min(float(w), self.cfg['max_symbol_weight']))
            capped[sym] = w
            total += w
        if total > self.cfg['max_gross_weight'] and total > 0:
            scale = self.cfg['max_gross_weight'] / total
            capped = {k: v*scale for k,v in capped.items()}
        # apply conserve gross scalar last
        capped = {k: v*gross_scalar for k,v in capped.items()}
        tot = sum(capped.values())
        return {k: v/tot for k,v in capped.items()} if tot>0 else {}
