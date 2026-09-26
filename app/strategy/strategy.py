"""
strategy.py

Contains the trading strategy used by the application.
Replace the placeholder implementation with your own strategy.
"""

import pandas as pd
import random

random.seed(42)

MIN_LENGTH = 0 # Warm-up offset: the first signal receives MIN_LENGTH + 1 candles.

def strategy(df : pd.DataFrame):
    """Return a random BUY, HOLD, or SELL signal for the backtest engine.

    Args:
        df: Historical OHLCV data through the current candle. This placeholder
            ignores the data and does not modify it.

    Returns:
        BUY requests a new LONG trade, SELL requests a new SHORT trade, and
        HOLD requests no new trade. The engine executes signals at the next
        candle's Open, subject to available cash.

    Notes:
        Signals do not close existing trades. The engine manages exits through
        each trade's take profit, stop loss, or the final backtest Close.
        The module seeds Python's shared random generator once at import.
        Later calls advance that generator; each backtest does not reset it.
        Replace this placeholder and set MIN_LENGTH for the strategy's warm-up.
    
    Example:
        def strategy(df)
            df = df.copy()
            fast_p = 20
            slow_p = 50
            sma_fast = df["Close"].ewm(span=fast_p, adjust=False).mean()
            sma_slow = df["Close"].ewm(span=slow_p, adjust=False).mean()
            
            if sma_fast[-2] <= sma_slow[-2] and sma_fast[-1] > sma_slow[-1]:
                return "BUY"   # Golden Cross -> Buy
            elif sma_fast[-2] >= sma_slow[-2] and sma_fast[-1] < sma_slow[-1]:
                return "SELL"  # Death Cross -> Sell
            return "HOLD"
    """
    return random.choice(["BUY","HOLD","SELL"])