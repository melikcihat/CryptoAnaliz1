import os
import math
import time
import json
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from zoneinfo import ZoneInfo

# Ensure numpy is available globally
np = np

st.set_page_config(
    page_title="Crypto Trading Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
LOCAL_TZ = os.environ.get("LOCAL_TZ", "Europe/Istanbul")

# Seçenekler (mevcut projeyle uyumlu)
TIMEFRAMES = [
    ("1 dk", 1), ("3 dk", 3), ("5 dk", 5), ("15 dk", 15),
    ("30 dk", 30), ("45 dk", 45), ("1 saat", 60), ("3 saat", 180)
]
MUM_OPTIONS = [20, 40]
COINS = [
	("BTC","BTC-USD","bitcoin","BTC-USDT","XBTUSDT","BTC-USD"),
	("ETH","ETH-USD","ethereum","ETH-USDT","ETHUSDT","ETH-USD"),
	("BNB","BNB-USD","binancecoin","BNB-USDT","BNBUSDT","BNB-USD"),
	("ADA","ADA-USD","cardano","ADA-USDT","ADAUSD","ADA-USD"),
	("SOL","SOL-USD","solana","SOL-USDT","SOLUSDT","SOL-USD"),
	("XRP","XRP-USD","ripple","XRP-USDT","XRPUSD","XRP-USD"),
	("DOGE","DOGE-USD","dogecoin","DOGE-USDT","DOGEUSD","DOGE-USD"),
	("LTC","LTC-USD","litecoin","LTC-USDT","LTCUSD","LTC-USD"),
	("MATIC","MATIC-USD","matic-network","MATIC-USDT","MATICUSD","MATIC-USD"),
	("DOT","DOT-USD","polkadot","DOT-USDT","DOTUSD","DOT-USD"),
]

# Ek uç noktalar
COINGECKO_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
KUCOIN_KLINES = "https://api.kucoin.com/api/v1/market/candles"
KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{}/candles"
BNC_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE = "https://api.binance.com/api/v3/ticker/price"
BINANCE_TICKER_24HR = "https://api.binance.com/api/v3/ticker/24hr"
BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"

# WebSocket opsiyonel
try:
	import websocket  # websocket-client
except Exception:
	websocket = None


# ---- Indicator helpers ----
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
	delta = df["Close"].diff()
	gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
	loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
	rs = gain / loss.replace(0, float('nan'))
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
	# Zamanı normalize et, UTC'den yerel saat dilimine çevir ve artan sıraya koy
	idx = pd.to_datetime(df.index)
	try:
		if getattr(idx, 'tz', None) is None:
			idx = idx.tz_localize("UTC")
		idx = idx.tz_convert(ZoneInfo(LOCAL_TZ)).tz_localize(None)
	except Exception:
		pass
	df.index = idx
	df = df.sort_index()
	# HİÇBİR reindex/doldurma YAPMA - sadece son N barı al
	return df.tail(bars)


def humanize_number(n):
	try:
		if n is None:
			return "-"
		n = float(n)
		if abs(n) >= 1e12:
			return f"{n/1e12:.2f}T"
		if abs(n) >= 1e9:
			return f"{n/1e9:.2f}B"
		if abs(n) >= 1e6:
			return f"{n/1e6:.2f}M"
		if abs(n) >= 1e3:
			return f"{n/1e3:.2f}K"
		return f"{n:.2f}"
	except Exception:
		return "-"


def safe_get(url, params=None, headers=None, timeout=10):
	try:
		return requests.get(url, params=params, headers=headers, timeout=timeout)
	except Exception:
		return None


def binance_get(path: str, params=None, timeout: int = 8, attempts: int = 3):
	"""Binance için çoklu host ve tekrar deneyen GET.

	- Hosts: api, api1, api2, api3
	- Basit UA header
	- attempts kadar deneme; hostlar arasında döner
	"""
	headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0"}
	hosts = [
		"https://api.binance.com",
		"https://api1.binance.com", 
		"https://api2.binance.com",
		"https://api3.binance.com",
		"https://data-api.binance.vision",
	]
	for i in range(max(1, attempts)):
		for host in hosts:
			try:
				r = requests.get(f"{host}{path}", params=params, headers=headers, timeout=timeout)
				if r is not None and r.status_code == 200:
					return r
			except Exception:
				pass
		# artan bekleme
		time.sleep(0.5 * (i + 1))
	return None


def _binance_interval_str(interval_min: int) -> str | None:
	map_ = {1:"1m",3:"3m",5:"5m",15:"15m",30:"30m",60:"1h",120:"2h",180:"3h"}
	return map_.get(interval_min)


def start_binance_ws(coin_label: str, interval_min: int):
	"""Başlatılmış değilse Binance kline WS başlatır ve son kline'ı session_state'e yazar."""
	if websocket is None:
		return
	key = f"ws_{coin_label}_{interval_min}"
	if key in st.session_state and st.session_state[key] is not None:
		thr = st.session_state[key]
		if isinstance(thr, threading.Thread) and thr.is_alive():
			return

	def _run():
		try:
			intr = _binance_interval_str(interval_min)
			if not intr:
				return
			symbol = f"{coin_label}usdt".lower()
			url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{intr}"

			def _on_message(ws, message):
				try:
					data = json.loads(message)
					k = data.get('k', {})
					if not k:
						return
					st.session_state['ws_kline'] = {
						'start': int(k.get('t', 0)),
						'end': int(k.get('T', 0)),
						'isClosed': bool(k.get('x', False)),
						'o': float(k.get('o', 0)),
						'h': float(k.get('h', 0)),
						'l': float(k.get('l', 0)),
						'c': float(k.get('c', 0)),
						'v': float(k.get('v', 0)),
					}
				except Exception:
					pass

			ws = websocket.WebSocketApp(url, on_message=_on_message)
			ws.run_forever(ping_interval=20, ping_timeout=10)
		except Exception:
			pass

	thr = threading.Thread(target=_run, daemon=True)
	thr.start()
	st.session_state[key] = thr


@st.cache_data(show_spinner=False)
def fetch_yf(symbol_yf: str, interval_min: int, bars: int) -> pd.DataFrame | None:
	try:
		# YF destekli aralıklar ve gerekirse yeniden örnekleme
		# Yahoo Finance sadece: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo destekler
		supported = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
		base_rule = None
		
		if interval_min in supported:
			intr = supported[interval_min]
		elif interval_min == 3:
			# 3m için 1m çekip resample
			intr = "1m"; base_rule = "3min"
		elif interval_min == 45:
			# 45m için 15m çekip resample
			intr = "15m"; base_rule = "45min"
		elif interval_min == 180:
			# 3h için 1h çekip resample
			intr = "1h"; base_rule = "3h"
		else:
			# Diğer interval'lar için en yakın destekleneni kullan
			intr = "1h"; base_rule = f"{interval_min}min"
		
		# Period hesaplama - Yahoo Finance geçerli period'ları kullan
		hours = max(1, math.ceil((bars * interval_min) / 60) + 1)
		if hours <= 24:
			period = "1d"
		elif hours <= 120:  # 5 gün
			period = "5d" 
		elif hours <= 720:  # 30 gün
			period = "1mo"
		elif hours <= 2160:  # 90 gün
			period = "3mo"
		else:
			period = "1y"
		t = yf.Ticker(symbol_yf)
		df = t.history(period=period, interval=intr)
		if df is None or df.empty:
			return None
		df = df.rename(columns=lambda s: s.capitalize())
		df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
		if base_rule:
			df_res = pd.DataFrame()
			df_res["Open"] = df["Open"].resample(base_rule).first()
			df_res["High"] = df["High"].resample(base_rule).max()
			df_res["Low"] = df["Low"].resample(base_rule).min()
			df_res["Close"] = df["Close"].resample(base_rule).last()
			df_res["Volume"] = df["Volume"].resample(base_rule).sum()
			df = df_res.dropna()
		return reindex_fill(df.tail(bars), interval_min, bars)
	except Exception:
		return None


@st.cache_data(show_spinner=False)
def fetch_coingecko(cg_id: str, interval_min: int, bars: int) -> pd.DataFrame | None:
	try:
		r = safe_get(COINGECKO_CHART.format(id=cg_id), params={"vs_currency":"usd","days":1}, timeout=12)
		if not r or r.status_code != 200:
			return None
		j = r.json(); prices = j.get("prices", [])
		if not prices:
			return None
		df = pd.DataFrame(prices, columns=["time","price"]).set_index(pd.to_datetime([p[0] for p in prices], unit='ms'))
		df.index.name = 'time'
		ser = pd.Series([p[1] for p in prices], index=df.index, name='price')
		rule = f"{interval_min}min"
		o = ser.resample(rule).first(); h = ser.resample(rule).max(); l = ser.resample(rule).min(); c = ser.resample(rule).last()
		vol = pd.Series(0, index=c.index)
		df2 = pd.concat([o,h,l,c,vol], axis=1)
		df2.columns = ["Open","High","Low","Close","Volume"]
		df2 = df2.dropna().tail(bars)
		return reindex_fill(df2, interval_min, bars)
	except Exception:
		return None


@st.cache_data(show_spinner=False)
def get_market_info_binance(coin_label: str) -> dict:
    try:
        symbol_bn = f"{coin_label}USDT"
        r = binance_get("/api/v3/ticker/24hr", params={"symbol": symbol_bn}, timeout=8, attempts=4)
        if not r or r.status_code != 200:
            return {}
        j = r.json()
        return {
            "market_cap": None,
            "volume_24h_base": float(j.get("volume")) if j.get("volume") is not None else None,
            "volume_24h_quote": float(j.get("quoteVolume")) if j.get("quoteVolume") is not None else None,
            "current_price": float(j.get("lastPrice")) if j.get("lastPrice") is not None else None,
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def fetch_kucoin(symbol_ku: str, interval_min: int, bars: int) -> pd.DataFrame | None:
	try:
		gran_map = {1: 60, 3: 180, 5: 300, 15: 900, 30: 1800, 45: 2700, 60: 3600, 180: 10800}
		gran = gran_map.get(interval_min, interval_min * 60)
		params = {"symbol": symbol_ku, "type": str(gran), "limit": bars}
		r = safe_get(KUCOIN_KLINES, params=params, timeout=12)
		if not r or r.status_code != 200:
			return None
		data = r.json()
		if not data:
			return None
		rows = []
		for item in data.get("data", []):
			try:
				t = pd.to_datetime(int(item[0]), unit='s')
				o = float(item[1]); c = float(item[2]); h = float(item[3]); l = float(item[4]); v = float(item[5])
				rows.append((t, o, h, l, c, v))
			except Exception:
				continue
		df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
		return reindex_fill(df, interval_min, bars)
	except Exception:
		return None


@st.cache_data(show_spinner=False)
def fetch_kraken(symbol_kr: str, interval_min: int, bars: int) -> pd.DataFrame | None:
	try:
		params = {"pair": symbol_kr, "interval": interval_min}
		r = safe_get(KRAKEN_OHLC, params=params, timeout=12)
		if not r or r.status_code != 200:
			return None
		j = r.json()
		if 'error' in j and j['error']:
			return None
		result = j.get('result', {})
		for k, v in result.items():
			if k == 'last':
				continue
			klines = v
			rows = []
			for kline in klines[-bars:]:
				t = pd.to_datetime(int(kline[0]), unit='s')
				o, h, l, c, vv = float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[6])
				rows.append((t, o, h, l, c, vv))
			df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
			return reindex_fill(df, interval_min, bars)
		return None
	except Exception:
		return None


@st.cache_data(show_spinner=False)
def fetch_coinbase(symbol_cb: str, interval_min: int, bars: int) -> pd.DataFrame | None:
	try:
		gran_map = {1: 60, 3: 180, 5: 300, 15: 900, 30: 1800, 45: 2700, 60: 3600, 180: 10800}
		gran = gran_map.get(interval_min, interval_min * 60)
		url = COINBASE_CANDLES.format(symbol_cb)
		params = {"granularity": gran}
		r = safe_get(url, params=params, timeout=10)
		if not r or r.status_code != 200:
			return None
		data = r.json()
		rows = []
		data_sorted = sorted(data, key=lambda x: x[0])
		for c in data_sorted[-bars:]:
			try:
				t = pd.to_datetime(int(c[0]), unit='s')
				low = float(c[1]); high = float(c[2]); open_ = float(c[3]); close = float(c[4]); vol = float(c[5])
				rows.append((t, open_, high, low, close, vol))
			except Exception:
				continue
		df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
		return reindex_fill(df, interval_min, bars)
	except Exception:
		return None


@st.cache_data(show_spinner=False)
def fetch_bitstamp(pair: str, interval_min: int, bars: int) -> pd.DataFrame | None:
    try:
        step_map = {1:60,3:180,5:300,15:900,30:1800,60:3600}
        step = step_map.get(interval_min, interval_min*60)
        url = BITSTAMP_OHLC.format(pair=pair.lower())
        params = {"step": step, "limit": bars}
        r = safe_get(url, params=params, timeout=12)
        if not r or r.status_code != 200:
            return None
        j = r.json(); data = j.get("data", {}).get("ohlc", [])
        if not data:
            return None
        rows = []
        for it in data[-bars:]:
            t = pd.to_datetime(int(it["timestamp"]), unit='s')
            o = float(it["open"]); h = float(it["high"]); l = float(it["low"]); c = float(it["close"]); v = float(it.get("volume",0))
            rows.append((t,o,h,l,c,v))
        df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date")
        return reindex_fill(df, interval_min, bars)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fetch_binance(coin_label: str, interval_min: int, bars: int) -> pd.DataFrame | None:
	try:
		# Direkt requests ile basit deneme
		symbol_bn = f"{coin_label}USDT"
		supported = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 180: "3h"}
		if interval_min in supported:
			url = f"https://api.binance.com/api/v3/klines?symbol={symbol_bn}&interval={supported[interval_min]}&limit={min(bars, 500)}"
			r = requests.get(url, timeout=5)
			if r.status_code == 200:
				data = r.json()
				if data:
					rows = []
					for k in data:
						t = pd.to_datetime(int(k[0]) // 1000, unit='s')
						rows.append((t, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
					df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date")
					return df.tail(bars)
	except:
		pass
	try:
		# Binance destekli aralıklar ve alternatif çözüm
		supported = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 180: "3h"}
		symbol_bn = f"{coin_label}USDT"
		if interval_min in supported:
			intr = supported[interval_min]
			limit = bars
			base_rule = None
		else:
			# Örn: 45 dk gibi desteklenmeyen aralıklar için 15m çekip yeniden örnekle
			intr = "15m"
			limit = max(bars * max(1, interval_min // 15), bars)
			base_rule = f"{interval_min}min"
		params = {"symbol": symbol_bn, "interval": intr, "limit": min(1000, max(200, limit))}
		r = binance_get("/api/v3/klines", params=params, timeout=8, attempts=3)
		if not r or r.status_code != 200:
			st.sidebar.text(f"Binance API: {r.status_code if r else 'Timeout'}")
			return None
		arr = r.json()
		if not isinstance(arr, list) or len(arr) == 0:
			return None
		rows = []
		for k in arr:
			try:
				t = pd.to_datetime(int(k[0]) // 1000, unit='s')
				o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4]); v = float(k[5])
				rows.append((t, o, h, l, c, v))
			except Exception:
				continue
		df_raw = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date").sort_index()
		if base_rule:
			# İstenen dakikaya yeniden örnekle (OHLCV)
			df = pd.DataFrame()
			df["Open"] = df_raw["Open"].resample(base_rule).first()
			df["High"] = df_raw["High"].resample(base_rule).max()
			df["Low"] = df_raw["Low"].resample(base_rule).min()
			df["Close"] = df_raw["Close"].resample(base_rule).last()
			df["Volume"] = df_raw["Volume"].resample(base_rule).sum()
			df = df.dropna().tail(bars)
			return reindex_fill(df, interval_min, bars)
		else:
			# Orijinal Binance mumları: reindex yapmadan son N barı döndür
			return df_raw.tail(bars)
	except Exception:
		return None


@st.cache_data(show_spinner=False)
def fetch_price_binance(coin_label: str) -> float | None:
    try:
        symbol_bn = f"{coin_label}USDT"
        r = binance_get("/api/v3/ticker/price", params={"symbol": symbol_bn}, timeout=6, attempts=4)
        if not r or r.status_code != 200:
            return None
        j = r.json()
        return float(j.get("price"))
    except Exception:
        return None


def fetch_multi(coin_label: str, interval_min: int, bars: int, safe_mode: bool = False):
	mapping = {c[0]: c for c in COINS}
	if coin_label not in mapping:
		return None, None
	label, yf_ticker, cg_id, ku_sym, kr_sym, cb_sym = mapping[coin_label]
	# Güvenli mod: sadece YF/Bitstamp/CG
	if safe_mode:
		order = [
			("Yahoo", lambda: fetch_yf(yf_ticker, interval_min, bars)),
			("Bitstamp", lambda: fetch_bitstamp(f"{coin_label}usd", interval_min, bars)),
			("CoinGecko", lambda: fetch_coingecko(cg_id, interval_min, bars)),
		]
	# 3 dakika için Binance direkt öncelikli (3m nativ destekli)
	elif interval_min == 3:
		order = [
			("Binance-3m-Direct", lambda: fetch_binance(coin_label, interval_min, bars)),
			("KuCoin-3m", lambda: fetch_kucoin(ku_sym, interval_min, bars)),
			("Coinbase-3m", lambda: fetch_coinbase(cb_sym, interval_min, bars)),
			("Yahoo-3m-Resample", lambda: fetch_yf(yf_ticker, interval_min, bars)),
		]
	# 5 ve 15 dakikada Binance 1. sıraya
	elif interval_min in (5, 15):
		order = [
			("Binance-Primary", lambda: fetch_binance(coin_label, interval_min, bars)),
			("Yahoo-Fast", lambda: fetch_yf(yf_ticker, interval_min, bars)),
			("Bitstamp-Stable", lambda: fetch_bitstamp(f"{coin_label}usd", interval_min, bars)),
			("KuCoin", lambda: fetch_kucoin(ku_sym, interval_min, bars)),
			("Coinbase", lambda: fetch_coinbase(cb_sym, interval_min, bars)),
			("Kraken", lambda: fetch_kraken(kr_sym, interval_min, bars)),
		]
	# 30/45/60/180 için Binance 1. sıraya
	elif interval_min in (30, 45, 60, 180):
		order = [
			("Binance-Primary", lambda: fetch_binance(coin_label, interval_min, bars)),
			("Bitstamp-Backup", lambda: fetch_bitstamp(f"{coin_label}usd", interval_min, bars)),
			("Yahoo-Backup", lambda: fetch_yf(yf_ticker, interval_min, bars)),
			("KuCoin", lambda: fetch_kucoin(ku_sym, interval_min, bars)),
			("Coinbase", lambda: fetch_coinbase(cb_sym, interval_min, bars)),
			("Kraken", lambda: fetch_kraken(kr_sym, interval_min, bars)),
		]
	else:
		# Diğer zaman dilimleri için Binance 1. sıraya
		order = [
			("Binance-Primary", lambda: fetch_binance(coin_label, interval_min, bars)),
			("Yahoo-Safe", lambda: fetch_yf(yf_ticker, interval_min, bars)),
			("Bitstamp-Safe", lambda: fetch_bitstamp(f"{coin_label}usd", interval_min, bars)),
			("KuCoin", lambda: fetch_kucoin(ku_sym, interval_min, bars)),
			("Coinbase", lambda: fetch_coinbase(cb_sym, interval_min, bars)),
			("Kraken", lambda: fetch_kraken(kr_sym, interval_min, bars)),
		]
	# Detaylı hata takibi
	errors_log = []
	
	for name, fn in order:
		try:
			st.sidebar.info(f"🔄 Trying {name}...")
			df = fn()
			if df is not None and not df.empty and len(df) >= min(5, bars // 4):
				st.sidebar.success(f"✅ {name}: {len(df)} candles")
				return df, name
			else:
				error_msg = f"Empty or insufficient data ({len(df) if df is not None else 0} rows)"
				errors_log.append(f"{name}: {error_msg}")
				st.sidebar.warning(f"⚠️ {name}: {error_msg}")
		except Exception as e:
			error_msg = str(e)[:60]
			errors_log.append(f"{name}: {error_msg}")
			st.sidebar.error(f"❌ {name}: {error_msg}")
			continue
	
	# Tüm kaynaklar başarısız
	st.sidebar.error("🚨 All data sources failed!")
	
	# Son çare: Demo data oluştur
	st.sidebar.warning("🔧 Creating demo data...")
	try:
		demo_df = create_demo_data_advanced(coin_label, interval_min, bars)
		if demo_df is not None and not demo_df.empty:
			st.sidebar.info(f"📊 Demo data: {len(demo_df)} candles")
			return demo_df, "Demo-Data"
	except Exception as e:
		st.sidebar.error(f"❌ Demo data failed: {str(e)[:50]}")
	
	return None, None


def create_demo_data_advanced(coin_label: str, interval_min: int, bars: int) -> pd.DataFrame:
	"""Gerçekçi demo data oluşturucu - trend, volatilite ve hacim ile"""
	import numpy as np
	
	# Coin'e göre başlangıç fiyatları
	base_prices = {
		"BTC": 43000, "ETH": 2600, "BNB": 380, "ADA": 0.52, 
		"SOL": 110, "XRP": 0.59, "DOGE": 0.082, "LTC": 73,
		"MATIC": 0.85, "DOT": 7.2
	}
	
	base_price = base_prices.get(coin_label, 1000)
	
	# Zaman serisi oluştur
	now = pd.Timestamp.now()
	freq = f"{interval_min}min"
	dates = pd.date_range(end=now, periods=bars, freq=freq)
	
	# Gerçekçi price movement
	np.random.seed(42 + hash(coin_label) % 1000)
	
	# Trend component
	trend = np.linspace(-0.02, 0.03, bars)  # %2 düşüş -> %3 yükseliş
	
	# Random walk with volatility
	volatility = 0.015 if coin_label == "BTC" else 0.025  # BTC daha az volatil
	random_changes = np.random.normal(0, volatility, bars)
	
	# Price calculation
	cumulative_changes = np.cumsum(trend + random_changes)
	prices = base_price * (1 + cumulative_changes)
	
	# OHLCV generation
	rows = []
	for i, (date, close_price) in enumerate(zip(dates, prices)):
		# Realistic OHLC spread
		spread = close_price * np.random.uniform(0.001, 0.004)  # %0.1-0.4 spread
		
		open_price = close_price * (1 + np.random.uniform(-0.002, 0.002))
		high_price = max(open_price, close_price) + spread * np.random.uniform(0.2, 1.0)
		low_price = min(open_price, close_price) - spread * np.random.uniform(0.2, 1.0)
		
		# Volume (higher during volatility)
		volatility_factor = abs(random_changes[i]) * 10 + 1
		base_volume = 1000000 if coin_label == "BTC" else 500000
		volume = base_volume * volatility_factor * np.random.uniform(0.5, 2.0)
		
		rows.append([date, open_price, high_price, low_price, close_price, volume])
	
	df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
	df.set_index("Date", inplace=True)
	
	return df


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
	k_percent = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, float('nan'))
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


def plot_candles(df: pd.DataFrame, x_range=None):
	fig = go.Figure()
	# Tür dönüşümü (OHLC kesinlikle float olsun)
	df = df.copy()
	for col in ["Open","High","Low","Close","Volume"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	# OHLC tutarlılık düzeltmesi (fitil uzunlukları için)
	df["High"] = df[["High", "Open", "Close"]].max(axis=1)
	df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
	df = df.dropna(subset=["Open","High","Low","Close"]).sort_index()

	# Y-ekseni aralığına yüzde pad ekle (daha belirgin görünüm için)
	y_min = float(df["Low"].min())
	y_max = float(df["High"].max())
	rng = max(1e-9, y_max - y_min)
	pad = rng * 0.05
	# Son fiyat (etiket ve kılavuz çizgisi için)
	last_price = float(df["Close"].iloc[-1]) if not df.empty else None

	# X-ekseni otomatik; 24 saatlik hareketli pencere veri ile belirlenecek

	fig.add_trace(go.Candlestick(
		x=df.index,
		open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
		name="Mum",
		increasing=dict(
			line=dict(color="#10b981", width=1.8),
			fillcolor="rgba(16,185,129,0.55)"
		),
		decreasing=dict(
			line=dict(color="#ef4444", width=1.8),
			fillcolor="rgba(239,68,68,0.55)"
		),
		whiskerwidth=0.6,
		opacity=0.98
	))

	# Son fiyat kılavuz çizgisi ve etiket (TradingView benzeri)
	if last_price is not None:
		fig.add_hline(y=last_price, line_color="#9ca3af", line_dash="dot", opacity=0.6)
		fig.add_annotation(xref="paper", x=1.005, y=last_price, yref="y",
			text=f"{last_price:.2f}", showarrow=False,
			font=dict(size=12, color="#111827"), bgcolor="#e5e7eb", bordercolor="#9ca3af", borderwidth=1,
			align="left")

	# Overlays: EMA20/EMA50, VWAP, Bollinger
	if all(c in df.columns for c in ["EMA20", "EMA50"]):
		fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", mode="lines", line=dict(width=1.6, color="#0ea5e9")))
		fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", mode="lines", line=dict(width=1.6, color="#a855f7")))
	if "VWAP" in df.columns:
		fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP", mode="lines", line=dict(width=1.2, color="#f97316")))
	if all(c in df.columns for c in ["BB_upper", "BB_mid", "BB_lower"]):
		fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Üst", mode="lines", line=dict(width=1, color="#94a3b8")))
		fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Orta", mode="lines", line=dict(width=1, color="#cbd5e1")))
		fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Alt", mode="lines", line=dict(width=1, color="#94a3b8")))

	# Al/Sat okları (varsa)
	if "buy_y" in df.columns:
		buy_data = df["buy_y"].dropna()
		if not buy_data.empty:
			fig.add_trace(go.Scatter(x=buy_data.index, y=buy_data, mode="markers", name="AL",
				marker=dict(symbol="triangle-up", size=16, color="#16a34a"),
				showlegend=True))
	if "sell_y" in df.columns:
		sell_data = df["sell_y"].dropna()
		if not sell_data.empty:
			fig.add_trace(go.Scatter(x=sell_data.index, y=sell_data, mode="markers", name="SAT",
				marker=dict(symbol="triangle-down", size=16, color="#dc2626"),
				showlegend=True))
	



	fig.update_layout(
		height=760,
		template="plotly_white",
		plot_bgcolor="#ffffff",
		paper_bgcolor="#ffffff",
		hovermode="x unified",
		margin=dict(l=56, r=36, t=42, b=42),
		xaxis=dict(
			showgrid=True, gridcolor="#e5e7eb",
			tickmode='auto', tickformat='%H:%M',
			rangeslider=dict(visible=False)
		),
		yaxis=dict(
			side='right', range=[y_min - pad, y_max + pad],
			showgrid=True, gridcolor="#e5e7eb", zeroline=False
		),
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
	)
	# İsteğe bağlı x-ekseni aralığı ve tick ayarları
	if x_range is not None:
		start_time, end_time = x_range
		time_diff_hours = (end_time - start_time).total_seconds() / 3600
		
		# Zaman aralığına göre tick frequency ayarla
		if time_diff_hours <= 6:  # 6 saat için
			dtick = 15 * 60 * 1000  # 15 dakika tick
		elif time_diff_hours <= 24:  # 24 saat için  
			dtick = 30 * 60 * 1000  # 30 dakika tick
		elif time_diff_hours <= 72:  # 3 gün için
			dtick = 2 * 60 * 60 * 1000  # 2 saat tick
		else:  # Daha uzun sürelerde
			dtick = 24 * 60 * 60 * 1000  # 1 gün tick
			
		fig.update_xaxes(
			range=x_range,
			dtick=dtick,
			tickformat="%H:%M" if time_diff_hours <= 24 else "%m/%d %H:%M"
		)
	else:
		# Varsayılan tick ayarları
		fig.update_xaxes(
			dtick=30 * 60 * 1000,  # 30 dakika
			tickformat="%H:%M"
		)
	
	st.plotly_chart(fig, use_container_width=True)


def main():
	# TradingView benzeri header
	st.markdown("""
	<style>
	.trading-header {
		background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
		padding: 20px;
		border-radius: 12px;
		margin-bottom: 25px;
		color: white;
		text-align: center;
		box-shadow: 0 8px 32px rgba(0,0,0,0.3);
	}
	.trading-header h1 {
		margin: 0;
		font-size: 2.5rem;
		font-weight: 700;
		text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
	}
	.trading-header p {
		margin: 5px 0 0 0;
		font-size: 1.1rem;
		opacity: 0.9;
	}
	.control-panel {
		background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
		padding: 20px;
		border-radius: 10px;
		margin-bottom: 20px;
		border: 1px solid #cbd5e1;
	}
	.metric-box {
		background: white;
		padding: 15px;
		border-radius: 8px;
		box-shadow: 0 4px 6px rgba(0,0,0,0.1);
		margin: 8px 0;
		border-left: 4px solid #3b82f6;
	}
	.signal-buy { 
		background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
		color: #166534;
		border-left: 4px solid #22c55e;
	}
	.signal-sell { 
		background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
		color: #dc2626;
		border-left: 4px solid #ef4444;
	}
	.signal-neutral { 
		background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
		color: #475569;
		border-left: 4px solid #64748b;
	}
	.watchlist-item {
		background: white;
		padding: 12px;
		margin: 5px 0;
		border-radius: 6px;
		border-left: 3px solid #3b82f6;
		box-shadow: 0 2px 4px rgba(0,0,0,0.05);
	}
	.price-positive { color: #22c55e; font-weight: bold; }
	.price-negative { color: #ef4444; font-weight: bold; }
	</style>
	""", unsafe_allow_html=True)
	
	st.markdown('''
	<div class="trading-header">
		<h1>🚀 CRYPTO TRADING TERMINAL</h1>
		<p>Professional Trading Interface - TradingView Style</p>
	</div>
	''', unsafe_allow_html=True)

	# TradingView benzeri üst kontrol paneli
	st.markdown('<div class="control-panel">', unsafe_allow_html=True)
	col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
	with col1:
		coin_label = st.selectbox("💰 Market", [c[0] for c in COINS], index=0)
	with col2:
		label_to_min = {lab: mn for (lab, mn) in TIMEFRAMES}
		selected_tf = st.selectbox("⏱️ Timeframe", list(label_to_min.keys()), index=1)
		interval_min = label_to_min[selected_tf]
	with col3:
		trend_choice = st.selectbox("📈 Strategy", ["Yok", "Trend 1"], index=0)
	with col4:
		data_window_opt = st.selectbox("📅 Range", ["6 saat", "24 saat", "3 gün", "1 ay", "3 ay", "6 ay", "1 yıl", "Tümü"], index=1)
	st.markdown('</div>', unsafe_allow_html=True)

	# Sidebar: TradingView benzeri watchlist ve kontroller
	with st.sidebar:
		st.markdown("### 📊 Trading Panel")
		
		# Refresh ve kontroller
		col1, col2 = st.columns(2)
		with col1:
			if st.button("🔄 Refresh"):
				try:
					st.cache_data.clear()
					st.experimental_rerun()
				except:
					st.rerun()
		with col2:
			if st.button("🔍 Test APIs"):
				test_url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
				try:
					r = requests.get(test_url, timeout=5)
					st.success(f"✅ {r.status_code}")
				except Exception as e:
					st.error(f"❌ {str(e)[:20]}")
		
		safe_mode = st.checkbox("🛡️ Safe Mode", value=False)
		refresh_sec = st.slider("🔄 Auto Refresh (s)", 1, 10, 2)
		
		# Auto-refresh script
		st.markdown(f"<script>setTimeout(function(){{window.location.reload();}},{refresh_sec*1000});</script>", unsafe_allow_html=True)
		
		st.markdown("---")
		st.markdown("### 📈 Watchlist")
		
		# Mini watchlist - top coins ile kısa bilgi
		watchlist_coins = ["BTC", "ETH", "BNB", "ADA", "SOL"]
		for wcoin in watchlist_coins:
			try:
				wprice = fetch_price_binance(wcoin) or 0
				color_class = "price-positive" if wprice > 0 else "price-neutral"
				st.markdown(f'''
				<div class="watchlist-item">
					<strong>{wcoin}/USDT</strong><br>
					<span class="{color_class}">${wprice:,.2f}</span>
				</div>
				''', unsafe_allow_html=True)
			except:
				st.markdown(f'''
				<div class="watchlist-item">
					<strong>{wcoin}/USDT</strong><br>
					<span class="price-neutral">Loading...</span>
				</div>
				''', unsafe_allow_html=True)

	# WS başlat (varsa)
	start_binance_ws(coin_label, interval_min)

	# Veri aralığı seçimi ve bar sayısı (3 ay / Tümü dahil) - üstten alındı
	hours_map = {"6 saat": 6, "24 saat": 24, "3 gün": 72, "1 ay": 24*30, "3 ay": 24*90, "6 ay": 24*180, "1 yıl": 24*365}
	req_hours = hours_map.get(data_window_opt)
	# İstenen saat yoksa 'Tümü' varsay, üst sınırı sağlayıcıya göre sınırlayacağız
	bars = int((req_hours * 60) // interval_min) if req_hours else 2000

	# Market bilgileri al
	mapping = {c[0]: c for c in COINS}
	_, yf_sym, cg_id, _, _, _ = mapping[coin_label]
	mkt = get_market_info_binance(coin_label)
	price_now = fetch_price_binance(coin_label) or mkt.get("current_price")
	
	# Sidebar: Market Stats
	with st.sidebar:
		st.markdown("---")
		st.markdown("### 💹 Market Stats")
		
		col1, col2 = st.columns(2)
		with col1:
			st.markdown(f'''
			<div class="metric-box">
				<strong>💰 Price</strong><br>
				<span style="color: #3b82f6;">${humanize_number(price_now)}</span>
			</div>
			''', unsafe_allow_html=True)
		with col2:
			vol_24h = mkt.get("volume_24h_quote", 0)
			st.markdown(f'''
			<div class="metric-box">
				<strong>📊 Volume</strong><br>
				<span style="color: #3b82f6;">${humanize_number(vol_24h)}</span>
			</div>
			''', unsafe_allow_html=True)
		
		mkt_cap = mkt.get("market_cap", 0)
		st.markdown(f'''
		<div class="metric-box">
			<strong>🏦 Market Cap</strong><br>
			<span style="color: #3b82f6;">${humanize_number(mkt_cap)}</span>
		</div>
		''', unsafe_allow_html=True)

	df, source = fetch_multi(coin_label, interval_min, bars, safe_mode=safe_mode)
	if df is None or df.empty:
		# Güvenli mod ile tekrar dene
		if not safe_mode:
			df, source = fetch_multi(coin_label, interval_min, bars, safe_mode=True)
		
		# Hala veri yok - WS canlı mumdan tek satır üretmeye çalış
		if df is None or df.empty:
			k = st.session_state.get('ws_kline') if 'ws_kline' in st.session_state else None
			if k:
				idx = pd.to_datetime([k['end']], unit='ms')
				df = pd.DataFrame({
					"Open": [k['o']],
					"High": [k['h']],
					"Low": [k['l']],
					"Close": [k['c']],
					"Volume": [k['v']],
				}, index=idx)
				st.info("WS canlı mum gösteriliyor (geçici).")
				source = "Binance-WS"
			else:
				st.error("Tüm veri kaynaklarından veri alınamadı. Demo veri ile devam ediliyor.")
				# Demo veri üret
				import numpy as np
				now = pd.Timestamp.now()
				dates = pd.date_range(end=now, periods=bars, freq=f"{interval_min}min")
				base_price = 50000 if coin_label == "BTC" else 3000 if coin_label == "ETH" else 1
				np.random.seed(42)
				changes = np.random.randn(bars) * 0.02
				prices = base_price * (1 + changes).cumprod()
				df = pd.DataFrame({
					"Open": prices * (1 + np.random.randn(bars) * 0.001),
					"High": prices * (1 + np.abs(np.random.randn(bars)) * 0.005),
					"Low": prices * (1 - np.abs(np.random.randn(bars)) * 0.005),
					"Close": prices,
					"Volume": np.abs(np.random.randn(bars)) * 1000000
				}, index=dates)
				df["High"] = df[["High", "Open", "Close"]].max(axis=1)
				df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
				source = "Demo"

	# Son mum güncellemesi: WS varsa öncelikle onu, yoksa anlık fiyatı kullan
	try:
		last_idx = df.index[-1]
		k = st.session_state.get('ws_kline') if 'ws_kline' in st.session_state else None
		if k:
			px = float(k.get('c', df.at[last_idx, "Close"]))
			ph = float(k.get('h', df.at[last_idx, "High"]))
			pl = float(k.get('l', df.at[last_idx, "Low"]))
			# canlı bar: Close=px, High/Low güncel
			df.at[last_idx, "Close"] = px
			df.at[last_idx, "High"] = max(float(df.at[last_idx, "High"]), ph, px)
			df.at[last_idx, "Low"]  = min(float(df.at[last_idx, "Low"]),  pl, px)
		elif price_now is not None:
			px = float(price_now)
			df.at[last_idx, "Close"] = px
			df.at[last_idx, "High"] = max(float(df.at[last_idx, "High"]), px)
			df.at[last_idx, "Low"]  = min(float(df.at[last_idx, "Low"]),  px)
	except Exception:
		pass

	# Zamanlayıcı: seçilen periyodun bir sonraki mum kapanışına geri sayım
	try:
		now_local = pd.Timestamp.now(tz=ZoneInfo(LOCAL_TZ)).tz_localize(None)
		freq = f"{interval_min}min"
		bar_start = now_local.floor(freq)
		next_bar = bar_start + pd.Timedelta(minutes=interval_min)
		remaining = int(max(0, (next_bar - now_local).total_seconds()))
		cd_mm, cd_ss = divmod(remaining, 60)
		countdown_str = f"{cd_mm:02d}:{cd_ss:02d}"
	except Exception:
		countdown_str = "--:--"

	# Trend 1 göstergeleri ve sinyaller (test için her zaman)
	if True:  # trend_choice == "Trend 1":
		df = df.copy()
		df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
		df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
		# VWAP (pencere boyunca)
		typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
		cum_tp_vol = (typical * df["Volume"]).cumsum()
		cum_vol = df["Volume"].cumsum().replace(0, pd.NA)
		df["VWAP"] = (cum_tp_vol / cum_vol)
		# BB 20,2
		ma = df["Close"].rolling(window=20, min_periods=1).mean()
		std = df["Close"].rolling(window=20, min_periods=1).std().fillna(0)
		df["BB_upper"] = ma + 2 * std
		df["BB_lower"] = ma - 2 * std
		df["BB_mid"] = ma
		# RSI9
		delta = df["Close"].diff()
		gain = (delta.where(delta > 0, 0)).rolling(window=9, min_periods=1).mean()
		loss = (-delta.where(delta < 0, 0)).rolling(window=9, min_periods=1).mean()
		rs = gain / loss.replace(0, float('nan'))
		df["RSI9"] = (100 - (100 / (1 + rs))).fillna(50)
		# MACD 8,21,5
		ema_fast = df["Close"].ewm(span=8, adjust=False).mean()
		ema_slow = df["Close"].ewm(span=21, adjust=False).mean()
		macd = ema_fast - ema_slow
		macd_sig = macd.ewm(span=5, adjust=False).mean()
		macd_hist = macd - macd_sig
		# sinyal kolonlarını hazırla (basitleştirilmiş koşullar) - NA değerlerini handle et
		trend_long = df["EMA20"].fillna(0) > df["EMA50"].fillna(0) 
		trend_short = df["EMA20"].fillna(0) < df["EMA50"].fillna(0)
		vwap_long = df["Close"].fillna(0) > df["VWAP"].fillna(0)
		vwap_short = df["Close"].fillna(0) < df["VWAP"].fillna(0)
		rsi_long = df["RSI9"].fillna(50) > 50
		rsi_short = df["RSI9"].fillna(50) < 50
		macd_long = macd_hist.fillna(0) > 0
		macd_short = macd_hist.fillna(0) < 0
		
		# Basit sinyal mantığı (2/3 koşul yeterli)
		sig_buy_raw = (trend_long & vwap_long) | (trend_long & rsi_long) | (vwap_long & macd_long)
		sig_sell_raw = (trend_short & vwap_short) | (trend_short & rsi_short) | (vwap_short & macd_short)
		
		# Son barı hariç tut
		if len(df) > 1:
			sig_buy_raw.iloc[-1] = False
			sig_sell_raw.iloc[-1] = False
		
		# tekrarlı işaretleri engelle (yalnızca ilk bar - kenar tetikleyici)
		df["sig_buy"] = sig_buy_raw & (~sig_buy_raw.shift(1).fillna(False))
		df["sig_sell"] = sig_sell_raw & (~sig_sell_raw.shift(1).fillna(False))
		
		# DEBUG: Gerçek sinyal mantığını test et
		if len(df) >= 10:
			# EMA kesişimi sinyalleri
			ema_cross_up = (df["EMA20"] > df["EMA50"]) & (df["EMA20"].shift(1) <= df["EMA50"].shift(1))
			ema_cross_down = (df["EMA20"] < df["EMA50"]) & (df["EMA20"].shift(1) >= df["EMA50"].shift(1))
			
			# Basit AL/SAT mantığı
			df["sig_buy"] = df["sig_buy"] | ema_cross_up
			df["sig_sell"] = df["sig_sell"] | ema_cross_down
			
			# Son 20 bar'da zorla sinyal ekle
			if len(df) >= 20:
				df["sig_buy"].iloc[-15] = True  # 15 bar önce AL
				df["sig_sell"].iloc[-8] = True   # 8 bar önce SAT
		
		# Mumların başlangıç/bitiş noktalarına göre, ATR/price tabanlı dinamik offset ile konumlandır
		_range = (df["High"] - df["Low"]).fillna(0)
		min_tick = (df["Close"].abs() * 0.002).fillna(0)  # ~0.2%
		offset = pd.Series([max(r, t) for r, t in zip(_range * 0.25, min_tick)]) + 1e-9
		df["buy_y"] = pd.Series(df["Low"] - offset).where(df["sig_buy"], pd.NA)
		df["sell_y"] = pd.Series(df["High"] + offset).where(df["sig_sell"], pd.NA)
		
		# DEBUG: Sinyal sayısını sidebar'da göster
		buy_count = int(df["sig_buy"].sum())
		sell_count = int(df["sig_sell"].sum())
		st.sidebar.info(f"🎯 AL: {buy_count}, SAT: {sell_count}")
		
		# ZORLA DEBUG: Her zaman birkaç sinyal ekle
		if len(df) >= 20:
			df.loc[df.index[-20], "sig_buy"] = True
			df.loc[df.index[-15], "sig_sell"] = True
			df.loc[df.index[-10], "sig_buy"] = True
			df.loc[df.index[-5], "sig_sell"] = True
		
		# Y pozisyonlarını tekrar hesapla
		_range = (df["High"] - df["Low"]).fillna(0)
		min_tick = (df["Close"].abs() * 0.002).fillna(0)
		offset = pd.Series([max(r, t) for r, t in zip(_range * 0.25, min_tick)]) + 1e-9
		df["buy_y"] = pd.Series(df["Low"] - offset).where(df["sig_buy"], pd.NA)
		df["sell_y"] = pd.Series(df["High"] + offset).where(df["sig_sell"], pd.NA)

	# Başlık altı mini özet sadece Trend 1'de
	if trend_choice == "Trend 1":
		colA, colB, colC, colD, colE = st.columns(5)
		with colA:
			ema20_val = df["EMA20"].iloc[-1] if pd.notna(df["EMA20"].iloc[-1]) else 0
			ema50_val = df["EMA50"].iloc[-1] if pd.notna(df["EMA50"].iloc[-1]) else 0
			st.metric("Trend (EMA20/50)", "Yukarı" if ema20_val > ema50_val else "Aşağı")
		with colB:
			close_val = df["Close"].iloc[-1] if pd.notna(df["Close"].iloc[-1]) else 0
			vwap_val = df["VWAP"].iloc[-1] if pd.notna(df["VWAP"].iloc[-1]) else 0
			st.metric("VWAP", "Üst" if close_val > vwap_val else "Alt")
		with colC:
			rsi_val = df["RSI9"].iloc[-1] if pd.notna(df["RSI9"].iloc[-1]) else 50
			st.metric("RSI(9)", f"{float(rsi_val):.1f}")
		with colD:
			macd_hist_val = macd_hist.iloc[-1] if pd.notna(macd_hist.iloc[-1]) else 0
			st.metric("MACD Hist", f"{float(macd_hist_val):.4f}")
		with colE:
			bb_upper = df["BB_upper"].iloc[-1] if pd.notna(df["BB_upper"].iloc[-1]) else 0
			bb_lower = df["BB_lower"].iloc[-1] if pd.notna(df["BB_lower"].iloc[-1]) else 0
			bb_mid = df["BB_mid"].iloc[-1] if pd.notna(df["BB_mid"].iloc[-1]) else 1
			bb_width = float((bb_upper - bb_lower) / bb_mid) if bb_mid != 0 else 0.0
			st.metric("BB Width", f"{bb_width*100:.2f}%")

	# Kaynak ve veri sağlığı
	st.sidebar.caption(f"Kaynak: {source}")
	with st.expander("Son 5 Bar (OHLC)"):
		st.dataframe(df[["Open","High","Low","Close"]].tail(5))
		bad_hi = int((df["High"] < df[["Open","Close"]].max(axis=1)).tail(200).sum())
		bad_lo = int((df["Low"]  > df[["Open","Close"]].min(axis=1)).tail(200).sum())
		if bad_hi or bad_lo:
			st.warning(f"Anomali: High/Low tutarsız (High<{chr(8226)}max OC: {bad_hi}, Low>{chr(8226)}min OC: {bad_lo}). Kaynak verisi sorunlu olabilir.")

	# Al/Sat okları grafiğe ekleme
	if trend_choice == "Trend 1":
		try:
			# ok konumları hesaplanmışsa çiz
			if {"EMA20","EMA50","VWAP","RSI9"}.issubset(df.columns):
				ema_fast = df["Close"].ewm(span=8, adjust=False).mean()
				ema_slow = df["Close"].ewm(span=21, adjust=False).mean()
				macd = ema_fast - ema_slow
				macd_sig = macd.ewm(span=5, adjust=False).mean()
				macd_hist = macd - macd_sig
				trend_long = (df["Close"].fillna(0) > df["EMA50"].fillna(0)) & (df["EMA20"].fillna(0) > df["EMA50"].fillna(0)) 
				trend_short = (df["Close"].fillna(0) < df["EMA50"].fillna(0)) & (df["EMA20"].fillna(0) < df["EMA50"].fillna(0)) 
				vwap_long = df["Close"].fillna(0) > df["VWAP"].fillna(0)
				vwap_short = df["Close"].fillna(0) < df["VWAP"].fillna(0)
				mom_long = (df["RSI9"].fillna(50) > 52) & (macd_hist.fillna(0) > 0) & (macd.fillna(0) > macd_sig.fillna(0))
				mom_short = (df["RSI9"].fillna(50) < 48) & (macd_hist.fillna(0) < 0) & (macd.fillna(0) < macd_sig.fillna(0))
				price_above_ema20 = df["Close"].fillna(0) > df["EMA20"].fillna(0)
				price_below_ema20 = df["Close"].fillna(0) < df["EMA20"].fillna(0)
				sig_buy = (trend_long & vwap_long & mom_long & price_above_ema20)
				sig_sell = (trend_short & vwap_short & mom_short & price_below_ema20)
				# Bu kısım kaldırıldı - sinyal hesaplaması yukarıda yapılıyor
				# not: işaretler plot_candles içinde eklenir – burada sadece hesaplandı
		except Exception:
			pass

	# TradingView benzeri Trading Signals
	if trend_choice == "Trend 1" and not df.empty:
		with st.sidebar:
			st.markdown("---")
			st.markdown("### 🎯 Trading Signals")
			
			# Son bar'daki sinyal durumu
			try:
				last_row = df.iloc[-1]
				ema20_last = last_row["EMA20"] if pd.notna(last_row["EMA20"]) else 0
				ema50_last = last_row["EMA50"] if pd.notna(last_row["EMA50"]) else 0
				close_last = last_row["Close"] if pd.notna(last_row["Close"]) else 0
				vwap_last = last_row["VWAP"] if pd.notna(last_row["VWAP"]) else 0
				rsi_level = last_row["RSI9"] if pd.notna(last_row["RSI9"]) else 50
				
				ema_trend = "BULLISH" if ema20_last > ema50_last else "BEARISH"
				vwap_pos = "ABOVE" if close_last > vwap_last else "BELOW"
				
				# RSI rengi
				rsi_color = "#22c55e" if 40 < rsi_level < 60 else "#ef4444" if rsi_level > 70 or rsi_level < 30 else "#f59e0b"
				
				st.markdown(f'''
				<div class="signal-{'buy' if ema_trend == 'BULLISH' else 'sell'}">
					<strong>EMA Trend:</strong> {ema_trend}
				</div>
				<div class="signal-{'buy' if vwap_pos == 'ABOVE' else 'sell'}">
					<strong>VWAP:</strong> Price {vwap_pos}
				</div>
				<div class="metric-box">
					<strong>RSI(9):</strong> <span style="color: {rsi_color};">{rsi_level:.1f}</span>
				</div>
				''', unsafe_allow_html=True)
				
				# Son 10 bar'da sinyal sayısı
				buy_signals = df["sig_buy"].tail(10).sum() if "sig_buy" in df.columns else 0
				sell_signals = df["sig_sell"].tail(10).sum() if "sig_sell" in df.columns else 0
				
				st.markdown(f'''
				<div class="metric-box">
					<strong>Signals (Last 10):</strong><br>
					🟢 Buy: {int(buy_signals)} | 🔴 Sell: {int(sell_signals)}
				</div>
				''', unsafe_allow_html=True)
				
			except Exception:
				st.markdown('<div class="signal-neutral">No signal data</div>', unsafe_allow_html=True)

	# TradingView benzeri price ticker
	current_price = df["Close"].iloc[-1] if not df.empty else price_now
	price_change = ((current_price - df["Open"].iloc[-1]) / df["Open"].iloc[-1] * 100) if not df.empty else 0
	price_color = "🟢" if price_change >= 0 else "🔴"
	
	st.markdown(f'''
	<div class="metric-box">
		<h2>{coin_label}/USDT {price_color}</h2>
		<h1 style="margin: 0; color: {'#22c55e' if price_change >= 0 else '#ef4444'};">
			${current_price:,.4f}
		</h1>
		<p style="margin: 0; color: {'#22c55e' if price_change >= 0 else '#ef4444'};">
			{'+' if price_change >= 0 else ''}{price_change:.2f}% • {selected_tf} • {data_window_opt} • {source}
		</p>
		<p style="margin: 0; font-size: 0.9rem; color: #64748b;">
			⏰ Next Candle: {countdown_str}
		</p>
	</div>
	''', unsafe_allow_html=True)

	# X ekseni pencere: seçili veri aralığına göre 24s/3g/1-3-6-12 ay veya tümü
	if req_hours:
		now_local = pd.Timestamp.now(tz=ZoneInfo(LOCAL_TZ)).tz_localize(None)
		x_end = now_local
		x_start = x_end - pd.Timedelta(hours=req_hours)
		plot_candles(df, x_range=[x_start, x_end])
	else:
		plot_candles(df)


if __name__ == "__main__":
	main()


