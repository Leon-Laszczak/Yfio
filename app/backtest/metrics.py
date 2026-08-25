"""Portfolio and risk metric utilities for historical backtest analysis.

This module computes a compact set of performance, volatility, drawdown,
and risk statistics from a time-indexed equity history and a trade log.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class MetricsComputer:
    """Calculate common investment metrics for a backtested strategy.

    The class expects a numeric price or equity series indexed by timestamps and a
    DataFrame of trades containing trade-level PnL data.
    """

    def __init__(self, history: pd.Series, trades: pd.DataFrame):
        """Initialize the metrics calculator.

        Args:
            history: Time-indexed price or equity history used to derive returns.
            trades: Trade records used to estimate win rate and related metrics.
                    Expected columns : type,amount, price_in, price_out, and pnl.
        """
        self.history = history
        self.trades = trades
        self.ppy = self.get_periods_per_year(self.history)
        self.ret = self.history.pct_change().dropna().to_numpy()

    def get_periods_per_year(self, hist: pd.Series) -> float:
        """Estimate the number of periods per year from the series duration.

        Args:
            hist: Historical equity or price series with a datetime index.

        Returns:
            Approximate periods per year based on the observation span.
        
        Raises:
            ValueError if history has less than two records.
        """

        if len(hist) < 2:
            raise ValueError("Insufficient amount of data")
        
        duration = hist.index[-1] - hist.index[0]
        self.years = duration.total_seconds() / (365.25 * 24 * 60 * 60)

        return len(hist) / self.years

    def compute_ratios(self, ret: np.array, ppy: float, cagr : float) -> tuple[float]:
        """Compute Sharpe, Sortino, and Calmar ratios from return data.

        Args:
            ret: Array of periodic returns.
            ppy: Estimated periods per year.

        Returns:
            A tuple containing the Sharpe, Sortino, and Calmar ratios.
        """
        mean = ret.mean()
        vol = ret.std()

        sharpe = cagr / (vol * np.sqrt(ppy)) if vol * np.sqrt(ppy) != 0 else 0.0

        mar = 0
        downside = np.minimum((ret - mar), 0)
        downside_deviation = np.sqrt(np.mean(downside**2))

        sortino = cagr / (downside_deviation * np.sqrt(ppy)) if downside_deviation * np.sqrt(ppy) != 0 else 0.0

        cum       = pd.Series((1 + ret).cumprod())
        roll_max  = cum.cummax()
        dd_series = (cum - roll_max) / roll_max
        max_dd    = dd_series.min()

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        return sharpe, sortino, calmar

    def compute_drawdown(self, ret: np.array) -> tuple:
        """Measure drawdown depth and recovery times.

        Args:
            ret: Array of periodic returns.

        Returns:
            A tuple containing maximum drawdown, maximum recovery time, and mean
            recovery time.
        """
        cum       = pd.Series((1 + ret).cumprod())
        roll_max  = cum.cummax()
        dd_series = (cum - roll_max) / roll_max
        max_dd    = dd_series.min()

        if max_dd < 0:
            through_idx = dd_series.idxmin()
            peak_val = roll_max[through_idx]
            peak_idx = cum[:through_idx][cum.loc[:through_idx] == peak_val].index[-1]
            rec_cand = cum[through_idx:][cum.loc[through_idx:] >= peak_val]
            rec_idx = rec_cand.index[0] if len(rec_cand > 0) else cum.index[-1]

            max_dur = cum.index[rec_idx]- cum.index[peak_idx]
        else:
            max_dur = 0

        durs = []

        for i in range(len(cum)):
            cur = 0
            r = i
            
            while r < len(cum) and cum[r] < roll_max[r]:
                cur += 1
                r += 1

            if 0 < r-1 < len(cum) and cum[r-1] < roll_max[r-1]:
                durs.append(cur)

        if len(durs) > 0:
            mean_dur = sum(durs) / len(durs)
        elif max_dur:
            mean_dur = sum(durs)/1
        else:
            mean_dur = 0.0

        return max_dd, max_dur, mean_dur

    def compute_var_and_cvar(self, ret: np.array) -> tuple[float]:
        """Estimate VaR and CVaR at 95% and 99% confidence levels.

        Args:
            ret: Array of periodic returns.

        Returns:
            A tuple containing VaR 95%, VaR 99%, CVaR 95%, and CVaR 99%.
        """
        var_95 = np.percentile(ret, 5)
        var_99 = np.percentile(ret, 1)

        cvar_95 = ret[ret <= np.percentile(ret, 5)].mean()
        cvar_99 = ret[ret <= np.percentile(ret, 1)].mean()

        return var_95, var_99, cvar_95, cvar_99

    def compute_pnl(self, history: pd.Series) -> tuple:
        """Compute absolute and percentage PnL over the series window.

        Args:
            history: Historical equity curve.

        Returns:
            A tuple containing total PnL and percentage PnL.
        """
        pnl = history.iloc[-1] - history.iloc[0]
        pct_pnl = pnl / history.iloc[0]

        return pnl, pct_pnl

    def compute_win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate the proportion of profitable trades.

        Args:
            trades: DataFrame containing trade-level PnL values.

        Returns:
            The win rate as a fraction between 0 and 1.
        """
        winining_trades = len(trades["pnl"][trades["pnl"] > 0])

        win_rate = winining_trades / len(trades) if len(trades) > 0 else 0.0

        return win_rate

    def compute_profit_factor(self, ret: np.array) -> float:
        """Compute the ratio of gross profit to gross loss.

        Args:
            ret: Array of periodic returns.

        Returns:
            The profit factor value.
        """
        profit_factor = ret[ret > 0].sum() / abs(ret[ret <= 0].sum())

        return profit_factor

    def compute_statistical(self, ret: np.array) -> tuple[float]:
        """Compute skewness and kurtosis of the return distribution.

        Args:
            ret: Array of periodic returns.

        Returns:
            A tuple containing skewness and kurtosis values.
        """
        skewness = skew(ret)
        kurtosis_ = kurtosis(ret)

        return skewness, kurtosis_

    def compute_metrics(self) -> dict:
        """Aggregate the main risk and performance metrics into one payload.

        Returns:
            A dictionary containing all computed backtest metrics.
        """
        cagr = (self.history.iloc[-1]/self.history.iloc[0]) ** (1/self.years)
        vol = self.ret.std() * np.sqrt(self.ppy)

        sharpe, sortino, calmar = self.compute_ratios(self.ret, self.ppy,cagr)
        max_dd, max_dur, mean_dur = self.compute_drawdown(self.ret)
        var_95, var_99, cvar_95, cvar_99 = self.compute_var_and_cvar(self.ret)
        pnl, pct_pnl = self.compute_pnl(self.history)
        win_rate = self.compute_win_rate(self.trades)
        profit_factor = self.compute_profit_factor(self.ret)
        skewness, kurtosis = self.compute_statistical(self.ret)

        payload = {
            "Total PnL": pnl,
            "Percent PnL": pct_pnl,
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_dd,
            "Max Recovery Time": max_dur,
            "Mean Recovery Time": mean_dur,
            "VaR 1d 95%": var_95,
            "VaR 1d 99%": var_99,
            "CVaR 1d 95%": cvar_95,
            "CVaR 1d 99%": cvar_99,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
        }

        return payload