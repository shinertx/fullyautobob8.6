import pandas as pd
from v26meme.data.regime import label_regime

def _evaluate_formula(row, formula):
    if not isinstance(formula[0], list):
        feature, op, value = formula
        try:
            if op == '>': return row[feature] > value
            if op == '<': return row[feature] < value
        except KeyError:
            return False
    left, logical_op, right = formula
    if logical_op == 'AND':
        return _evaluate_formula(row, left) and _evaluate_formula(row, right)
    else:
        return _evaluate_formula(row, left) or _evaluate_formula(row, right)

class SimLab:
    def __init__(self, fees_bps, slippage_bps):
        self.fee = fees_bps/10000.0
        self.slippage = slippage_bps/10000.0

    def _stats(self, returns: pd.Series) -> dict:
        if returns.empty: return {"n_trades": 0}
        equity = (1+returns).cumprod()
        dd = (equity - equity.cummax()) / equity.cummax()
        down_std = returns[returns < 0].std()
        return {
            "n_trades": int(returns.shape[0]),
            "win_rate": float((returns > 0).mean()),
            "avg_return": float(returns.mean()),
            "sortino": float(returns.mean()/down_std) if (down_std is not None and down_std>0) else 0.0,
            "sharpe": float(returns.mean()/returns.std()) if (returns.std() is not None and returns.std()>0) else 0.0,
            "mdd": float(dd.min()),
            "returns": [float(x) for x in returns.tolist()],
        }

    def run_backtest(self, df: pd.DataFrame, formula: list) -> dict:
        if df.empty or len(df) < 10: return {}
        df = df.copy()
        df['regime'] = label_regime(df)
        signals = df.apply(lambda row: _evaluate_formula(row, formula), axis=1)
        edges = signals.astype(int).diff().fillna(0)
        # Handle initial in-position (signal True on first bar) so we open a trade at start
        in_trade, entry_price = False, 0.0
        if bool(signals.iloc[0]):
            in_trade = True
            entry_price = df['close'].iloc[0] * (1 + self.slippage)
        trades, regimes = [], []
        for i in range(len(df)):
            if edges.iloc[i] > 0 and not in_trade:
                in_trade, entry_price = True, df['close'].iloc[i] * (1 + self.slippage)
            elif edges.iloc[i] < 0 and in_trade:
                in_trade = False
                exit_price = df['close'].iloc[i] * (1 - self.slippage)
                trades.append(((exit_price-entry_price)/entry_price) - (2*self.fee))
                regimes.append(df['regime'].iloc[i])
        if in_trade:
            exit_price = df['close'].iloc[-1] * (1 - self.slippage)
            trades.append(((exit_price-entry_price)/entry_price) - (2*self.fee))
            regimes.append(df['regime'].iloc[-1])

        if not trades: return {"all": {"n_trades": 0}}

        ser = pd.Series(trades)
        res = {"all": self._stats(ser)}
        rser = pd.Series(trades, index=regimes)
        for r in list(dict.fromkeys(regimes)):
            res[r] = self._stats(rser[rser.index == r])
        return res
