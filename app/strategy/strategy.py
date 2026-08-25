"""
strategy.py

Contains the trading strategy used by the application.
Replace the placeholder implementation with your own strategy.
"""

import pandas as pd
import random

def strategy(df : pd.DataFrame):
    """
    Determine the trading action based on historical OHLCV data.

    This is currently a placeholder implementation that randomly selects
    a trading action. Replace it with an actual trading strategy.

    Args:
        df: A pandas DataFrame containing historical OHLCV data.
        Expected columns are Open, High, Low, Close, and Volume.
    
    Returns:
        The trading action to execute.. Supported values:
        BUY - Open a LONG position or close a SHORT position, 
        HOLD - do nothing, 
        SELL - Close a LONG postion or open a SHORT position
    
    Notes:
        This implementation is only a placeholder and makes random trading
        decisions. It should be replaced with an actual trading strategy.

        If your strategy requires a minimum DataFrame length the function should
        return a tuple containing a signal and a minimum DataFrame length
    
    Example:
        >>> action = strategy(df)
        >>> print(action)
        "BUY"

        >>> action = strategy(df)
        >>> print(action)
        ("BUY",20)
    """
    return random.choice(["BUY","HOLD","SELL"])