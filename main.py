"""
main.py

Dashboard for running the backtest.

"""
from app.backtest.backtest import backtest

import streamlit as st
import pandas as pd

def run_backtest(ticker,period,interval,transaction_cost,position_size,position_size_base,take_profit,stop_loss):
    """Convert percentage inputs to fractions and run the selected backtest.

    Save successful results in Streamlit session state. Display an error on
    failure while keeping any previously stored results.
    """
    with st.spinner("Running backtest..."):
        try:
            payload,history,trades = backtest(ticker,period,interval,transaction_cost/100,position_size/100,position_size_base,take_profit/100,stop_loss/100)
            st.session_state.payload = payload
            st.session_state.history = history
            st.session_state.trades = trades
        except Exception as e:
            st.error(f"An error occured during the backtest. Details: {str(e)}")

required_keys = ("payload","history","trades")
if not all(key in st.session_state for key in required_keys):
    st.session_state.payload = {
            "Total PnL": 0,
            "Percent PnL": 0,
            "CAGR": 0,
            "Volatility": 0,
            "Sharpe Ratio": 0,
            "Sortino Ratio": 0,
            "Calmar Ratio": 0,
            "Max Drawdown": 0,
            "Max Recovery Time": 0,
            "Mean Recovery Time": 0,
            "VaR 1d 95%": 0,
            "VaR 1d 99%": 0,
            "CVaR 1d 95%": 0,
            "CVaR 1d 99%": 0,
            "Win Rate": 0,
            "Profit Factor": 0,
            "Skewness": 0,
            "Kurtosis": 0,
        }
    st.session_state.history = pd.Series()
    st.session_state.trades = pd.DataFrame({"type":[],"amount" : [], "entry_price":[],"close_price" : [], "pnl" : []})


st.title("Yfio - Backtesting Engine")

ticker = st.text_input("Input Stock Symbol")
period = st.selectbox("Input time range for backtest", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],5)
interval = st.selectbox("Input candle interval", ["1m","2m","5m","15m","30m","60m","90m","1h","4h","1d","5d","1wk","1mo","3mo"],9)

with st.expander("Advanced Options"):
    transaction_cost = st.slider("Input transaction cost (in %)",0.0,1.0,0.1,0.01)
    position_size = st.slider("Input position size (in %)",0.0,100.0,10.0,0.1)
    position_size_base = st.selectbox("Select position size base", ["initial", "equity", "cash"], 0)
    take_profit = st.slider("Input take profit (in %)",0.0,100.0,10.0,0.1)
    stop_loss = st.slider("Input stop loss (in %)",0.0,100.0,10.0,0.1)

run_disabled = ticker.strip() == "" # Running the backtest is disabled until the user inputs the ticker

st.button("Run backtest", on_click=run_backtest,args=(ticker,period,interval,transaction_cost,position_size,position_size_base,take_profit,stop_loss),disabled=run_disabled)

st.header("Metrics")
st.metric("Total PnL",f"{st.session_state.payload['Total PnL']:.4f}")
st.metric("% PnL",f"{st.session_state.payload['Percent PnL'] * 100:.4f}%")
st.metric("Sharpe Ratio",f"{st.session_state.payload['Sharpe Ratio']:.4f}")
st.metric("Max Drawdown",f"{st.session_state.payload['Max Drawdown'] * 100:.4f}%")
st.metric("Win Rate",f"{st.session_state.payload['Win Rate'] * 100:.4f}%")

with st.expander("See All Metrics"):
    st.table(st.session_state.payload)

st.header("Equity Curve")
st.line_chart(st.session_state.history)

st.header("Trade records")
st.write(st.session_state.trades)