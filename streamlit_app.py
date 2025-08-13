import os
import math
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Analiz (Web)", layout="wide")

# Seçenekler (mevcut projeyle uyumlu)
TIMEFRAMES = [("3 dk", 3), ("5 dk", 5), ("15 dk", 15)]
MUM_OPTIONS = [20, 40]
COINS = [
	("BTC", "BTC-USD"),
	("ETH", "ETH-USD"),
	("BNB", "BNB-USD"),
	("ADA", "ADA-USD"),
	("SOL", "SOL-USD"),
	("XRP", "XRP-USD"),
	("DOGE", "DOGE-USD"),
	("LTC", "LTC-USD"),
	("MATIC", "MATIC-USD"),
	("DOT", "DOT-USD"),
]


# ---- Indicator helpers ----
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
	delta = df["Close"].diff()
	gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
	loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
	rs = gain / loss.replace(0, np.nan)
	rsi = 100 - (100 / (1 + rs))
	return rsi.fillna(50)


def calculate_macd(df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
	ema_short = df["Close"].ewm(span=short_window, adjust=False).mean()
	ema_long = df["Close"].ewm(span=long_window, adjust=False).mean()
	macd = ema_short - ema_long
	macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
	macd_hist = macd - macd_signal
	return macd, macd_signal, macd_hist


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2):
	ma = df["Close"].rolling(window=window, min_periods=1).mean()
	std = df["Close"].rolling(window=window, min_periods=1).std().fillna(0)
	upper = ma + num_std * std
	lower = ma - num_std * std
	return upper, lower, ma


def calculate_ema(close: pd.Series, span: int) -> pd.Series:
	return close.ewm(span=span, adjust=False).mean()


def reindex_fill(df: pd.DataFrame, interval_min: int, bars: int) -> pd.DataFrame:
	if df is None or df.empty:
		return df
	df = df.copy()
	df.index = pd.to_datetime(df.index)
	freq = f"{interval_min}min"
	end = df.index.max()
	start = end - pd.Timedelta(minutes=interval_min * (bars - 1))
	new_idx = pd.date_range(start=start, end=end, freq=freq)
	df = df.reindex(new_idx)
	df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill().bfill()
	df["Volume"] = df["Volume"].fillna(0)
	return df.tail(bars)


@st.cache_data(show_spinner=False)
def fetch_yf(symbol_yf: str, interval_min: int, bars: int) -> pd.DataFrame | None:
	try:
		intr = {3: "3m", 5: "5m", 15: "15m"}.get(interval_min, f"{interval_min}m")
		hours = max(1, math.ceil((bars * interval_min) / 60) + 1)
		period = f"{hours}h"
		t = yf.Ticker(symbol_yf)
		df = t.history(period=period, interval=intr, progress=False)
		if df is None or df.empty:
			return None
		df = df.rename(columns=lambda s: s.capitalize())
		df = df[["Open", "High", "Low", "Close", "Volume"]].tail(bars).copy()
		return reindex_fill(df, interval_min, bars)
	except Exception:
		return None


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
	# RSI
	df["RSI"] = calculate_rsi(df, period=14)
	# MACD
	macd, macd_signal, macd_hist = calculate_macd(df, 12, 26, 9)
	df["MACD"] = macd
	df["MACD_signal"] = macd_signal
	df["MACD_hist"] = macd_hist
	# Bollinger
	bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(df, window=20, num_std=2)
	df["BB_upper"], df["BB_lower"], df["BB_mid"] = bb_upper, bb_lower, bb_mid
	# EMAs
	df["EMA50"] = calculate_ema(df["Close"], 50)
	df["EMA200"] = calculate_ema(df["Close"], 200)
	# Stochastic (basitleştirilmiş)
	low_min = df["Low"].rolling(window=14, min_periods=1).min()
	high_max = df["High"].rolling(window=14, min_periods=1).max()
	k_percent = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
	k_percent = k_percent.fillna(50)
	d_percent = k_percent.rolling(window=3, min_periods=1).mean().fillna(50)
	df["STOCH_K"], df["STOCH_D"] = k_percent, d_percent

	# Sinyaller
	df["sig_rsi_buy"] = df["RSI"] < 30
	df["sig_rsi_sell"] = df["RSI"] > 70
	df["sig_macd_buy"] = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)
	df["sig_macd_sell"] = (df["MACD_hist"] < 0) & (df["MACD_hist"].shift(1) >= 0)
	df["sig_bb_buy"] = (df["Close"] <= df["BB_lower"] * 1.002)
	df["sig_bb_sell"] = (df["Close"] >= df["BB_upper"] * 0.998)
	df["sig_ema_bull"] = df["EMA50"] > df["EMA200"]
	df["sig_ema_bear"] = df["EMA50"] < df["EMA200"]
	df["sig_stoch_buy"] = (df["STOCH_K"] < 20) & (df["STOCH_K"].shift(1) < df["STOCH_D"].shift(1)) & (df["STOCH_K"] > df["STOCH_D"])
	df["sig_stoch_sell"] = (df["STOCH_K"] > 80) & (df["STOCH_K"].shift(1) > df["STOCH_D"].shift(1)) & (df["STOCH_K"] < df["STOCH_D"])

	df["sig_buy_all"] = (
		df["sig_rsi_buy"] &
		df["sig_macd_buy"].fillna(False) &
		df["sig_bb_buy"].fillna(False) &
		df["sig_ema_bull"].fillna(False) &
		df["sig_stoch_buy"].fillna(False)
	)

	df["sig_sell_all"] = (
		df["sig_rsi_sell"] &
		df["sig_macd_sell"].fillna(False) &
		df["sig_bb_sell"].fillna(False) &
		df["sig_ema_bear"].fillna(False) &
		df["sig_stoch_sell"].fillna(False)
	)

	return df


def plot_candles(df: pd.DataFrame):
	fig = go.Figure()
	fig.add_trace(go.Candlestick(
		x=df.index,
		open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
		name="Mum"
	))
	# EMAs
	if "EMA50" in df and "EMA200" in df:
		fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", mode="lines"))
		fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", mode="lines"))
	# Bollinger
	if "BB_upper" in df and "BB_lower" in df:
		fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Üst", mode="lines", line=dict(width=1)))
		fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Orta", mode="lines", line=dict(width=1)))
		fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Alt", mode="lines", line=dict(width=1)))
	# İşaretler
	try:
		buy_idx = df.index[df["sig_buy_all"] == True]
		sell_idx = df.index[df["sig_sell_all"] == True]
		if len(buy_idx):
			fig.add_trace(go.Scatter(
				x=buy_idx,
				y=(df.loc[buy_idx, "Low"] * 0.995),
				mode="markers",
				marker_symbol="triangle-up",
				marker_size=12,
				name="BUY"
			))
		if len(sell_idx):
			fig.add_trace(go.Scatter(
				x=sell_idx,
				y=(df.loc[sell_idx, "High"] * 1.005),
				mode="markers",
				marker_symbol="triangle-down",
				marker_size=12,
				name="SELL"
			))
	except Exception:
		pass

	fig.update_layout(height=650, xaxis_rangeslider_visible=False)
	st.plotly_chart(fig, use_container_width=True)


def main():
	st.title("Crypto Analiz (Web)")

	col1, col2, col3 = st.columns(3)
	with col1:
		coin_label = st.selectbox("Coin", [c[0] for c in COINS], index=0)
		coin_map = {c[0]: c[1] for c in COINS}
		symbol = coin_map[coin_label]
	with col2:
		label_to_min = {lab: mn for (lab, mn) in TIMEFRAMES}
		selected_tf = st.selectbox("Zaman Dilimi", list(label_to_min.keys()), index=1)
		interval_min = label_to_min[selected_tf]
	with col3:
		bars = st.selectbox("Mum", MUM_OPTIONS, index=0)

	df = fetch_yf(symbol, interval_min, bars)
	if df is None or df.empty:
		st.warning("Veri alınamadı. İnternet/VPN/Firewall kontrol et.")
		return

	# Göstergeler ve sinyaller
	df = build_signals(df)

	# Son sinyal bilgisi
	last_sig = "—"
	last_row = df.iloc[-1]
	if bool(last_row.get("sig_buy_all")):
		last_sig = "BUY (ALL)"
	elif bool(last_row.get("sig_sell_all")):
		last_sig = "SELL (ALL)"

	st.subheader(f"{coin_label} | {selected_tf} | {bars} mum — Son Sinyal: {last_sig}")
	plot_candles(df)


if __name__ == "__main__":
	main()


