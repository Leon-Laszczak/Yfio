"""Portfolio and risk metric utilities for historical backtest analysis.

This module computes a compact set of performance, volatility, drawdown,
and risk statistics from a time-indexed equity history and a trade log.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

MIN_DRAWDOWN = -0.025 # noise filter: ignore drawdown episodes shallower than 2.5% so tiny fluctuations aren't counted as separate "drawdowns"

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
        
        if not isinstance(history.index,pd.DatetimeIndex):
            raise TypeError("Index in history should be a datetime index")

        self.history = history
        self.trades = trades
        self.ppy = self.get_periods_per_year()
        self.ret = self.history.pct_change().dropna().to_numpy()
        self.daily_ret = self.get_daily_returns()

    def get_periods_per_year(self) -> float:
        """Estimate the number of periods per year from the series duration.

        Args:
            hist: Historical equity or price series with a datetime index.

        Returns:
            Approximate periods per year based on the observation span.
        
        Raises:
            ValueError if history has less than two records.
        """

        if len(self.history) < 2:
            raise ValueError("Insufficient amount of data")
        
        duration = self.history.index[-1] - self.history.index[0]
        self.years = duration.total_seconds() / (365.25 * 24 * 60 * 60)

        return (len(self.history) - 1) / self.years
    
    def get_daily_returns(self) -> np.ndarray:
        """
        Convert the equity curve to daily returns.

        The last portfolio value from each calendar day is used.
        Days without observations are ignored.
        """
        if not isinstance(self.history.index, pd.DatetimeIndex):
            raise TypeError("History must have a DatetimeIndex")

        daily_equity = (
            self.history
            .sort_index()
            .resample("1D")
            .last()
            .dropna()
        )

        daily_returns = daily_equity.pct_change().dropna()

        return daily_returns.to_numpy()

    def compute_ratios(self, cagr : float) -> tuple[float]:
        """Compute Sharpe, Sortino, and Calmar ratios from return data.

        Args:
            ret: Array of periodic returns.
            ppy: Estimated periods per year.

        Returns:
            A tuple containing the Sharpe, Sortino, and Calmar ratios.
        """
        arthmetic_return = self.ret.mean() * self.ppy
        vol = self.ret.std()

        sharpe = arthmetic_return / (vol * np.sqrt(self.ppy)) if vol * np.sqrt(self.ppy) != 0 else 0.0

        mar = 0
        downside = np.minimum((self.ret - mar), 0)
        downside_deviation = np.sqrt(np.mean(downside**2))

        sortino = arthmetic_return / (downside_deviation * np.sqrt(self.ppy)) if downside_deviation * np.sqrt(self.ppy) != 0 else 0.0

        cum = pd.Series(
            np.concatenate(([1.0], (1 + self.daily_ret).cumprod()))
        )
        roll_max  = cum.cummax()
        dd_series = (cum - roll_max) / roll_max
        max_dd    = dd_series.min()

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        return sharpe, sortino, calmar

    def compute_drawdown(self) -> tuple:
        """Measure drawdown depth and recovery times.

        Args:
            ret: Array of periodic returns.

        Returns:
            A tuple containing maximum drawdown, maximum recovery time, and mean
            recovery time.
        """

        cum = pd.Series(
                np.concatenate(([1.0], (1 + self.daily_ret).cumprod()))
            )
        roll_max  = cum.cummax()
        dd_series = (cum - roll_max) / roll_max
        max_dd    = dd_series.min()

        if max_dd < 0:
            through_idx = dd_series.idxmin()
            peak_val = roll_max[through_idx]
            peak_idx = cum[:through_idx][cum.loc[:through_idx] == peak_val].index[-1]
            rec_cand = cum[through_idx:][cum.loc[through_idx:] >= peak_val]
            rec_idx = rec_cand.index[0] if len(rec_cand) > 0 else cum.index[-1]

            max_dur = (cum.index[rec_idx]- cum.index[peak_idx])
        else:
            max_dur = 0

        recovery_times = []
        drawdown = cum/roll_max - 1
        in_dd = False
        peak_idx = None
        trough_val = 0.0

        for i in range(len(cum)):
            idx = i
            dd = drawdown.iloc[i]

            if dd < MIN_DRAWDOWN and not in_dd:
                in_dd = True
                peak_idx = cum.index[i - 1] if i > 0 else idx
                trough_val = dd

            elif dd < 0 and in_dd:
                if dd < trough_val:
                    trough_val = dd

            elif dd >= 0 and in_dd:
                in_dd = False
                trough_val = 0
                recovery_times.append((idx-peak_idx))

        if len(recovery_times) > 0:
            mean_dur = sum(recovery_times) / len(recovery_times)
        else:
            mean_dur = 0.0

        return max_dd, max_dur, mean_dur

    def compute_var_and_cvar(self) -> tuple[float]:
        """Estimate VaR and CVaR at 95% and 99% confidence levels.

        Args:
            ret: Array of periodic returns.

        Returns:
            A tuple containing VaR 95%, VaR 99%, CVaR 95%, and CVaR 99%.
        """
        daily_ret = self.daily_ret[np.isfinite(self.daily_ret)]

        if daily_ret.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        var_95 = np.percentile(self.daily_ret, 5)
        var_99 = np.percentile(self.daily_ret, 1)

        cvar_95 = self.daily_ret[self.daily_ret <= np.percentile(self.daily_ret, 5)].mean()
        cvar_99 = self.daily_ret[self.daily_ret <= np.percentile(self.daily_ret, 1)].mean()

        return var_95, var_99, cvar_95, cvar_99

    def compute_pnl(self) -> tuple:
        """Compute absolute and percentage PnL over the series window.

        Args:
            history: Historical equity curve.

        Returns:
            A tuple containing total PnL and percentage PnL.
        """
        pnl = self.history.iloc[-1] - self.history.iloc[0]
        pct_pnl = pnl / self.history.iloc[0]

        return pnl, pct_pnl

    def compute_win_rate(self) -> float:
        """Calculate the proportion of profitable trades.

        Args:
            trades: DataFrame containing trade-level PnL values.

        Returns:
            The win rate as a fraction between 0 and 1.
        """
        winining_trades = len(self.trades["pnl"][self.trades["pnl"] > 0])

        win_rate = winining_trades / len(self.trades) if len(self.trades) > 0 else 0.0

        return win_rate

    def compute_profit_factor(self) -> float:
        """Compute the ratio of gross profit to gross loss.

        Args:
            ret: Array of periodic returns.

        Returns:
            The profit factor value.
        """
        pnl = self.trades["pnl"]
        gross_profit = pnl[pnl>0].sum()
        gross_loss = pnl[pnl<=0].sum()

        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0.0

        if gross_loss == 0 and gross_profit > 0:
            return float("inf")

        return profit_factor

    def compute_statistical(self) -> tuple[float]:
        """Compute skewness and kurtosis of the return distribution.

        Args:
            ret: Array of periodic returns.

        Returns:
            A tuple containing skewness and kurtosis values.
        """
        skewness = skew(self.ret)
        kurtosis_ = kurtosis(self.ret)

        return skewness, kurtosis_

    def compute_metrics(self) -> dict:
        """Aggregate the main risk and performance metrics into one payload.

        Returns:
            A dictionary containing all computed backtest metrics.
        """
        cagr = (self.history.iloc[-1]/self.history.iloc[0]) ** (1/self.years) - 1
        vol = self.ret.std() * np.sqrt(self.ppy)

        sharpe, sortino, calmar = self.compute_ratios(cagr)
        max_dd, max_dur, mean_dur = self.compute_drawdown()
        var_95, var_99, cvar_95, cvar_99 = self.compute_var_and_cvar()
        pnl, pct_pnl = self.compute_pnl()
        win_rate = self.compute_win_rate()
        profit_factor = self.compute_profit_factor()
        skewness, kurtosis = self.compute_statistical()

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