"""
main.py

Dashboard for running the backtest.

"""
from app.backtest.backtest import backtest

import streamlit as st
import pandas as pd

def run_backtest(ticker,period,interval,transaction_cost):
    with st.spinner("Running backtest..."):
        payload,history,trades = backtest(ticker,period,interval,transaction_cost/100)
        st.session_state.payload = payload
        st.session_state.history = history
        st.session_state.trades = trades

if ("payload" and "history" and "trades") not in st.session_state:
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
transaction_cost = st.slider("Input transaction cost (in %)",0.0,1.0,0.1,0.01)

run_diasabled = ticker.strip() == "" # Running the backtest is diasabled until the user inputs the ticker

st.button("Run backtest", on_click=run_backtest,args=(ticker,period,interval,transaction_cost),disabled=run_diasabled)

st.header("Metrics")
st.metric("Total PnL",st.session_state.payload["Total PnL"])
st.metric("Sharpe Ratio",st.session_state.payload["Sharpe Ratio"])
st.metric("Max Drawdown",st.session_state.payload["Max Drawdown"])
st.metric("Win Rate",st.session_state.payload["Win Rate"])

with st.expander("See All Metrics"):
    st.table(st.session_state.payload)

st.header("Equity Curve")
st.line_chart(st.session_state.history)

st.header("Trade records")
st.write(st.session_state.trades)