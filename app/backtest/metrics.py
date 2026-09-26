"""Portfolio and risk metric utilities for historical backtest analysis.

This module computes a compact set of performance, volatility, drawdown,
and risk statistics from a time-indexed equity history and a trade log.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Threshold for mean recovery episodes only; maximum drawdown uses all declines.
MIN_DRAWDOWN = -0.025

class MetricsComputer:
    """Calculate common investment metrics for a backtested strategy.

    The class expects a numeric price or equity series indexed by timestamps and a
    DataFrame of trades containing trade-level PnL data.
    """

    def __init__(self, history: pd.Series, trades: pd.DataFrame):
        """Initialize metrics from an equity history and closed-trade PnL.

        Args:
            history: Numeric series with a DatetimeIndex, at least two observations,
                and a nonzero observation span. The engine supplies portfolio equity.
            trades: Closed-trade DataFrame with a pnl column. Trade metrics use this
                column only; the engine supplies PnL after entry and exit fees.
        """
        
        if not isinstance(history.index,pd.DatetimeIndex):
            raise TypeError("Index in history should be a datetime index")

        self.history = history
        self.trades = trades
        self.ppy = self.get_periods_per_year()
        self.ret = self.history.pct_change().dropna().to_numpy()
        self.daily_ret = self.get_daily_returns()

    def get_periods_per_year(self) -> float:
        """Estimate return periods per year from self.history's observation span.

        Store the elapsed calendar time in self.years using 365.25 days per year.
        Divide the number of return intervals by that span rather than assuming a
        fixed trading calendar.

        Returns:
            Estimated return periods per year.

        Raises:
            ValueError: If history contains fewer than two observations.
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
        """Compute Sharpe, Sortino, and Calmar ratios from stored returns.

        Args:
            cagr: Annualized compound return, expressed as a fraction.

        Returns:
            Sharpe, Sortino, and Calmar ratios, in that order. Sharpe and Sortino
            use periodic returns and estimated periods per year, with a zero
            risk-free rate and zero minimum acceptable return. Calmar divides
            cagr by the absolute maximum drawdown of daily returns. A zero
            denominator produces 0.0 for the corresponding ratio.
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
        """Measure drawdown and recovery durations from stored daily returns.

        Returns:
            Maximum drawdown as a non-positive fraction, duration of the deepest
            drawdown, and mean duration of completed episodes below MIN_DRAWDOWN.
            Durations count observed daily-return steps, not elapsed calendar days.

        Notes:
            The payload calls the second value "Max Recovery Time", but this is
            the recovery duration of the deepest drawdown, not necessarily the
            longest duration. If it never recovers, the duration ends at the last
            observation. Mean recovery starts one step before the threshold breach
            and excludes episodes that remain unrecovered. No completed qualifying
            episode produces a mean of 0.0.
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
        """Compute historical tail-return statistics from stored daily returns.

        Returns:
            The 5th and 1st return percentiles (VaR 95% and 99%), followed by the
            mean returns at or below each threshold (CVaR 95% and 99%). Values are
            signed return fractions, not positive loss amounts. If no finite daily
            return exists, return four zeros.
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
        """Compute the change between the first and last stored equity values.

        Returns:
            Absolute PnL and fractional PnL relative to starting equity. The
            "Percent PnL" payload value is a fraction, not a value multiplied by 100.
        """
        pnl = self.history.iloc[-1] - self.history.iloc[0]
        pct_pnl = pnl / self.history.iloc[0]

        return pnl, pct_pnl

    def compute_win_rate(self) -> float:
        """Calculate the fraction of closed trades with strictly positive PnL.

        Returns:
            Profitable trades divided by all closed trades. Break-even trades
            count in the denominator. Return 0.0 when no trades exist.
        """
        winining_trades = len(self.trades["pnl"][self.trades["pnl"] > 0])

        win_rate = winining_trades / len(self.trades) if len(self.trades) > 0 else 0.0

        return win_rate

    def compute_profit_factor(self) -> float:
        """Divide summed positive trade PnL by absolute summed non-positive PnL.

        The engine's trade PnL already includes entry and exit transaction fees.

        Returns:
            Profit factor, infinity for profits with no losses, or 0.0 when neither
            profits nor losses exist.
        """
        pnl = self.trades["pnl"]
        gross_profit = pnl[pnl>0].sum()
        gross_loss = pnl[pnl<=0].sum()

        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0.0

        if gross_loss == 0 and gross_profit > 0:
            return float("inf")

        return profit_factor

    def compute_statistical(self) -> tuple[float]:
        """Compute skewness and excess kurtosis of stored periodic returns.

        Returns:
            SciPy's default skewness and Fisher excess kurtosis estimates. These
            use the original observation interval, not resampled daily returns.
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