# crypto_app_candles_full.py
# Binance API kaldırıldı, diğer kaynaklar ve isteklerin doğrultusunda düzenlendi.
# 5 göstergeli entegrasyon: RSI, MACD, EMA50/EMA200, Bollinger Bands, Stochastic

import os
import sys
import time
import math
import threading
import requests
import pandas as pd
import numpy as np
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf

import matplotlib
matplotlib.use("TkAgg")
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Windows beep
try:
    import winsound
    WINSOUND = True
except Exception:
    WINSOUND = False

REFRESH_SECONDS = 30
TIMEFRAMES = [("3 dk", 3), ("5 dk", 5), ("15 dk", 15)]
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

COINGECKO_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
COINGECKO_STATUS = "https://api.coingecko.com/api/v3/coins/{id}/status_updates"
KUCOIN_KLINES = "https://api.kucoin.com/api/v1/market/candles"
KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{}/candles"

# ---- Utilities ----
def beep_alert(n=2):
    if WINSOUND:
        try:
            for _ in range(n):
                winsound.Beep(1000, 350)
                time.sleep(0.15)
        except Exception:
            pass
    else:
        # fallback: print (no sound)
        print("!! ALARM !!")

def safe_get(url, params=None, headers=None, timeout=10):
    try:
        return requests.get(url, params=params, headers=headers, timeout=timeout)
    except Exception:
        return None

def reindex_fill(df, interval_min, bars):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    freq = f"{interval_min}min"
    end = df.index.max()
    start = end - pd.Timedelta(minutes=interval_min*(bars-1))
    new_idx = pd.date_range(start=start, end=end, freq=freq)
    df = df.reindex(new_idx)
    # fill OHLC forward/backwards; volume fill 0
    df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].ffill().bfill()
    df['Volume'] = df['Volume'].fillna(0)
    return df.tail(bars)

# ---- Data fetchers ----
def fetch_coingecko(cg_id, interval_min, bars):
    try:
        r = safe_get(COINGECKO_CHART.format(id=cg_id), params={"vs_currency":"usd","days":1}, timeout=12)
        if not r or r.status_code != 200:
            return None
        j = r.json()
        prices = j.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["time","price"])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        rule = f"{interval_min}min"
        o = df['price'].resample(rule).first()
        h = df['price'].resample(rule).max()
        l = df['price'].resample(rule).min()
        c = df['price'].resample(rule).last()
        vol = pd.Series(0,index=c.index)
        df2 = pd.concat([o,h,l,c,vol], axis=1)
        df2.columns = ["Open","High","Low","Close","Volume"]
        df2 = df2.dropna().tail(bars)
        return reindex_fill(df2, interval_min, bars)
    except Exception:
        return None

def fetch_kucoin(symbol_ku, interval_min, bars):
    try:
        gran_map = {3:180,5:300,15:900}
        gran = gran_map.get(interval_min, interval_min*60)
        params = {"symbol": symbol_ku, "type": str(gran), "limit": bars}
        r = safe_get(KUCOIN_KLINES, params=params, timeout=8)
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
                rows.append((t,o,h,l,c,v))
            except Exception:
                continue
        df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date")
        return reindex_fill(df, interval_min, bars)
    except Exception:
        return None

def fetch_kraken(symbol_kr, interval_min, bars):
    try:
        params = {"pair": symbol_kr, "interval": interval_min}
        r = safe_get(KRAKEN_OHLC, params=params, timeout=8)
        if not r or r.status_code != 200:
            return None
        j = r.json()
        if 'error' in j and j['error']:
            return None
        result = j.get('result', {})
        for k,v in result.items():
            if k == 'last': continue
            klines = v
            rows = []
            for kline in klines[-bars:]:
                t = pd.to_datetime(int(kline[0]), unit='s')
                o,h,l,c,vv = float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[6])
                rows.append((t,o,h,l,c,vv))
            df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date")
            return reindex_fill(df, interval_min, bars)
        return None
    except Exception:
        return None

def fetch_coinbase(symbol_cb, interval_min, bars):
    try:
        gran_map = {3:180,5:300,15:900}
        gran = gran_map.get(interval_min, interval_min*60)
        url = COINBASE_CANDLES.format(symbol_cb)
        params = {"granularity": gran}
        r = safe_get(url, params=params, timeout=10)
        if not r or r.status_code != 200:
            return None
        data = r.json()
        rows = []
        # ensure sorted ascending by timestamp
        data_sorted = sorted(data, key=lambda x: x[0])
        for c in data_sorted[-bars:]:
            try:
                t = pd.to_datetime(int(c[0]), unit='s')
                low = float(c[1]); high = float(c[2]); open_ = float(c[3]); close = float(c[4]); vol = float(c[5])
                rows.append((t,open_,high,low,close,vol))
            except Exception:
                continue
        df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"]).set_index("Date")
        return reindex_fill(df, interval_min, bars)
    except Exception:
        return None

def fetch_yfinance(symbol_yf, interval_min, bars):
    try:
        intr = {3:"3m",5:"5m",15:"15m"}.get(interval_min, f"{interval_min}m")
        hours = max(1, math.ceil((bars * interval_min) / 60) + 1)
        period = f"{hours}h"
        t = yf.Ticker(symbol_yf)
        df = t.history(period=period, interval=intr, progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=lambda s: s.capitalize())
        df = df[["Open","High","Low","Close","Volume"]].tail(bars).copy()
        return reindex_fill(df, interval_min, bars)
    except Exception:
        return None

def get_market_info(cg_id):
    try:
        r = safe_get(COINGECKO_MARKETS, params={"vs_currency":"usd","ids":cg_id}, timeout=8)
        if not r or r.status_code != 200:
            return {}
        arr = r.json()
        if not arr:
            return {}
        it = arr[0]
        return {
            "market_cap": it.get("market_cap"),
            "volume_24h": it.get("total_volume"),
            "current_price": it.get("current_price")
        }
    except Exception:
        return {}

def get_news(cg_id):
    try:
        r = safe_get(COINGECKO_STATUS.format(id=cg_id), timeout=8)
        if not r or r.status_code != 200:
            return []
        j = r.json()
        updates = j.get("status_updates", [])
        res = []
        for u in updates[:5]:
            res.append({
                "date": u.get("created_at"),
                "title": u.get("user_title") or u.get("title") or "",
                "desc": u.get("description") or ""
            })
        return res
    except Exception:
        return []

# ---- Indicators ----
def calculate_rsi(df, period=14):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    if df is None or df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    ema_short = df['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(df, window=20, num_std=2):
    if df is None or df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ma = df['Close'].rolling(window=window, min_periods=1).mean()
    std = df['Close'].rolling(window=window, min_periods=1).std().fillna(0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower

def calculate_ema(df, span):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df['Close'].ewm(span=span, adjust=False).mean()

def calculate_stochastic(df, k_window=14, d_window=3):
    if df is None or df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    low_min = df['Low'].rolling(window=k_window, min_periods=1).min()
    high_max = df['High'].rolling(window=k_window, min_periods=1).max()
    k_percent = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    k_percent = k_percent.fillna(50)
    d_percent = k_percent.rolling(window=d_window, min_periods=1).mean().fillna(50)
    return k_percent, d_percent


def calculate_vwap(df):
	if df is None or df.empty:
		return pd.Series(dtype=float)
	typical = (df['High'] + df['Low'] + df['Close']) / 3.0
	cum_tp_vol = (typical * df['Volume']).cumsum()
	cum_vol = df['Volume'].cumsum().replace(0, np.nan)
	return (cum_tp_vol / cum_vol).fillna(method='bfill').fillna(method='ffill')


def calculate_atr(df, period=14):
	if df is None or df.empty:
		return pd.Series(dtype=float)
	high = df['High']
	low = df['Low']
	close = df['Close']
	prev_close = close.shift(1)
	tr1 = (high - low).abs()
	tr2 = (high - prev_close).abs()
	tr3 = (low - prev_close).abs()
	true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
	atr = true_range.ewm(span=period, adjust=False).mean()
	return atr.fillna(method='bfill').fillna(method='ffill')

def calc_levels(df, window):
    if df is None or df.empty:
        return None
    tail = df.tail(window)
    if tail.empty:
        return None
    sup = float(tail["Low"].min())
    res = float(tail["High"].max())
    limit_sup = float(tail["Low"].mean())
    limit_res = float(tail["High"].mean())
    rng = max(1e-9, res - sup)
    up_target = res + rng * 0.5
    down_target = sup - rng * 0.5
    return {
        "support": sup,
        "resistance": res,
        "limit_support": limit_sup,
        "limit_resistance": limit_res,
        "up_target": up_target,
        "down_target": down_target
    }

# ---- App Class ----
class CryptoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crypto Analiz - Lokal Uygulama")
        self.root.geometry("1250x780")

        self.coin_var = tk.StringVar(value=COINS[0][0])
        self.tf_var = tk.IntVar(value=5)
        self.mum_var = tk.IntVar(value=20)
        self.latest_df = None
        self.latest_levels = None
        self.latest_info = {}
        self.latest_news = []
        self.canvas1 = None
        self.canvas2 = None
        self.stop_flag = threading.Event()

        self._build_ui()
        self.update_now()

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side="top", fill="x", padx=8, pady=6)

        ttk.Label(top, text="Coin:").pack(side="left", padx=(4,2))
        self.coin_combo = ttk.Combobox(top, values=[c[0] for c in COINS], textvariable=self.coin_var, state="readonly", width=10)
        self.coin_combo.pack(side="left")
        self.coin_combo.bind("<<ComboboxSelected>>", lambda e: self.on_change())

        tf_frame = ttk.Frame(top); tf_frame.pack(side="left", padx=10)
        ttk.Label(tf_frame, text="Zaman Dilimi:").pack(side="left")
        for lab, mn in TIMEFRAMES:
            r = ttk.Radiobutton(tf_frame, text=lab, value=mn, variable=self.tf_var, command=self.on_change)
            r.pack(side="left", padx=3)

        ms = ttk.Frame(top); ms.pack(side="left", padx=10)
        ttk.Label(ms, text="Mum:").pack(side="left")
        for m in MUM_OPTIONS:
            rb = ttk.Radiobutton(ms, text=f"{m} mum", value=m, variable=self.mum_var, command=self.on_change)
            rb.pack(side="left", padx=3)

        ctrl = ttk.Frame(top); ctrl.pack(side="right")
        self.refresh_btn = ttk.Button(ctrl, text="Yenile", command=self.update_now)
        self.refresh_btn.pack(side="left", padx=4)
        self.auto_btn = ttk.Button(ctrl, text="Otomatik Başlat", command=self.toggle_auto)
        self.auto_btn.pack(side="left", padx=4)

        main = ttk.Frame(self.root); main.pack(fill="both", expand=True, padx=8, pady=6)
        left_frame = ttk.Frame(main); left_frame.pack(side="left", fill="both", expand=True)
        right_frame = ttk.Frame(main, width=360); right_frame.pack(side="right", fill="y")

        # 1. Bölüm - Al-Sat Sinyalleri ve Grafik
        ttk.Label(left_frame, text="1. Bölüm: Al-Sat Sinyalleri ve Grafik", font=("Arial", 12, "bold")).pack(pady=6)
        self.plot_frame1 = ttk.Frame(left_frame)
        self.plot_frame1.pack(fill="both", expand=True)

        # 2. Bölüm - Sadece Mum Grafiği
        ttk.Label(left_frame, text="2. Bölüm: Sadece Mum Grafiği", font=("Arial", 12, "bold")).pack(pady=(20,6))
        self.plot_frame2 = ttk.Frame(left_frame)
        self.plot_frame2.pack(fill="both", expand=True)

        # Bilgiler ve Haberler
        ttk.Label(right_frame, text="Bilgiler & Haberler", font=("Arial", 12, "bold")).pack(pady=(6,4))
        self.info_txt = tk.Text(right_frame, width=44, height=30, wrap="word")
        self.info_txt.pack(padx=6, pady=4)
        self.info_txt.config(state="disabled")
        self.news_btn = ttk.Button(right_frame, text="Haberleri Göster", command=self.show_news)
        self.news_btn.pack(pady=6)

        self.status = ttk.Label(self.root, text="Hazır", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def set_status(self, txt):
        self.status.config(text=txt)

    def toggle_auto(self):
        if hasattr(self, "auto_thread") and self.auto_thread.is_alive():
            self.stop_flag.set()
            self.auto_btn.config(text="Otomatik Başlat")
            self.set_status("Otomatik durduruldu.")
        else:
            self.stop_flag.clear()
            self.auto_thread = threading.Thread(target=self._auto_loop, daemon=True)
            self.auto_thread.start()
            self.auto_btn.config(text="Otomatik Durdur")
            self.set_status("Otomatik başlatıldı.")

    def _auto_loop(self):
        while not self.stop_flag.is_set():
            self.update_now()
            for _ in range(REFRESH_SECONDS):
                if self.stop_flag.is_set():
                    break
                time.sleep(1)

    def on_change(self):
        self.update_now()

    def update_now(self):
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        coin_label = self.coin_var.get()
        mapping = {c[0]:c for c in COINS}
        if coin_label not in mapping:
            self.set_status("Coin bulunamadı.")
            return
        label, yf_ticker, cg_id, ku_sym, kr_sym, cb_sym = mapping[coin_label]
        tf = self.tf_var.get()
        bars = self.mum_var.get()
        self.set_status(f"Veri çekiliyor: {label} | {tf} dk | {bars} mum ...")

        df = None
        source = None
        try_order = [
            ("CoinGecko", lambda: fetch_coingecko(cg_id, tf, bars)),
            ("KuCoin", lambda: fetch_kucoin(ku_sym, tf, bars)),
            ("Kraken", lambda: fetch_kraken(kr_sym, tf, bars)),
            ("Coinbase", lambda: fetch_coinbase(cb_sym, tf, bars)),
            ("Yahoo", lambda: fetch_yfinance(yf_ticker, tf, bars))
        ]
        for name, func in try_order:
            try:
                df = func()
                if df is not None and not df.empty:
                    source = name
                    break
            except Exception:
                df = None
        if df is None or df.empty:
            self.set_status("Veri alınamadı. İnternet/VPN/Firewall kontrol et.")
            self._write_info("Veri alınamadı. Lütfen bağlantınızı kontrol edin.")
            return

        # ensure datetime index and proper columns
        df.index = pd.to_datetime(df.index)
        df = df[["Open","High","Low","Close","Volume"]].copy()

        # ---------- ADD: Indicators calculation ----------
        # RSI genel (14)
        df['RSI'] = calculate_rsi(df, period=14)
        # Trend1 (5dk): EMA20/EMA50, VWAP, RSI(9), MACD(8,21,5), BB(20,2)
        if tf == 5:
            try:
                df['EMA20_T1'] = calculate_ema(df, 20)
                df['EMA50_T1'] = calculate_ema(df, 50)
                df['VWAP_T1'] = calculate_vwap(df)
                macd_t1, macd_sig_t1, macd_hist_t1 = calculate_macd(df, short_window=8, long_window=21, signal_window=5)
                df['MACD_T1'] = macd_t1
                df['MACDsig_T1'] = macd_sig_t1
                df['MACDhist_T1'] = macd_hist_t1
                bb_u_t1, bb_l_t1 = calculate_bollinger_bands(df, window=20, num_std=2)
                df['BBu_T1'] = bb_u_t1
                df['BBl_T1'] = bb_l_t1
                df['RSI9_T1'] = calculate_rsi(df, period=9)
                # ATR tabanlı SL/Taker seviyeleri için ATR(14)
                df['ATR14_T1'] = calculate_atr(df, period=14)
            except Exception:
                pass

        # MACD
        macd, macd_signal, macd_hist = calculate_macd(df, 12, 26, 9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist

        # Bollinger Bands
        bb_upper, bb_lower = calculate_bollinger_bands(df, window=20, num_std=2)
        df['BB_upper'] = bb_upper
        df['BB_lower'] = bb_lower
        df['BB_mid'] = (bb_upper + bb_lower) / 2

        # EMAs
        df['EMA50'] = calculate_ema(df, 50)
        df['EMA200'] = calculate_ema(df, 200)

        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(df, k_window=14, d_window=3)
        df['STOCH_K'] = stoch_k
        df['STOCH_D'] = stoch_d

        # Generate individual signals
        df['sig_rsi_buy'] = df['RSI'] < 30
        df['sig_rsi_sell'] = df['RSI'] > 70
        df['sig_macd_buy'] = (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)
        df['sig_macd_sell'] = (df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) >= 0)
        df['sig_bb_buy'] = (df['Close'] <= df['BB_lower'] * 1.002)  # close near lower band
        df['sig_bb_sell'] = (df['Close'] >= df['BB_upper'] * 0.998)  # close near upper band
        df['sig_ema_bull'] = df['EMA50'] > df['EMA200']
        df['sig_ema_bear'] = df['EMA50'] < df['EMA200']
        df['sig_stoch_buy'] = (df['STOCH_K'] < 20) & (df['STOCH_K'].shift(1) < df['STOCH_D'].shift(1)) & (df['STOCH_K'] > df['STOCH_D'])
        df['sig_stoch_sell'] = (df['STOCH_K'] > 80) & (df['STOCH_K'].shift(1) > df['STOCH_D'].shift(1)) & (df['STOCH_K'] < df['STOCH_D'])

        # Combined signals: require most indicators to agree
        df['sig_buy_all'] = (
            df['sig_rsi_buy'] &
            df['sig_macd_buy'].fillna(False) &
            df['sig_bb_buy'].fillna(False) &
            df['sig_ema_bull'].fillna(False) &
            df['sig_stoch_buy'].fillna(False)
        )
        df['sig_sell_all'] = (
            df['sig_rsi_sell'] &
            df['sig_macd_sell'].fillna(False) &
            df['sig_bb_sell'].fillna(False) &
            df['sig_ema_bear'].fillna(False) &
            df['sig_stoch_sell'].fillna(False)
        )
        # ---------- END: Indicators ----------

        levels = calc_levels(df, bars)
        info = get_market_info(cg_id)
        news = get_news(cg_id)

        self.latest_df = df
        self.latest_levels = levels
        self.latest_info = info
        self.latest_news = news

        self.root.after(0, self._update_gui)
        self.set_status(f"Veri güncellendi ({source}).")

    def _update_gui(self):
        # If no data, skip
        if self.latest_df is None or self.latest_df.empty:
            self._write_info("Grafik için yeterli veri yok.")
            return
        self._write_info(self._format_info(self.latest_levels, self.latest_info, self.latest_df))
        # Trend 1 özet paneli (sadece 5 dk iken)
        try:
            if self.tf_var.get() == 5:
                last = self.latest_df.iloc[-1]
                trend_text = []
                # === Trend 1 kuralları (Pine eşleştirme) ===
                # Filtreler
                trend_long = bool(last.get('Close') > last.get('EMA50_T1') and last.get('EMA20_T1') > last.get('EMA50_T1')) if not last.isna().any() else False
                trend_short = bool(last.get('Close') < last.get('EMA50_T1') and last.get('EMA20_T1') < last.get('EMA50_T1')) if not last.isna().any() else False
                vwap_long = bool(last.get('Close') > last.get('VWAP_T1')) if not pd.isna(last.get('VWAP_T1')) else False
                vwap_short = bool(last.get('Close') < last.get('VWAP_T1')) if not pd.isna(last.get('VWAP_T1')) else False
                macd_hist_pos = bool(last.get('MACDhist_T1') > 0)
                macd_above_sig = bool(last.get('MACD_T1') > last.get('MACDsig_T1'))
                mom_long = bool(last.get('RSI9_T1') > 52 and macd_hist_pos and macd_above_sig) if not pd.isna(last.get('RSI9_T1')) else False
                mom_short = bool(last.get('RSI9_T1') < 48 and (not macd_hist_pos) and (not macd_above_sig)) if not pd.isna(last.get('RSI9_T1')) else False
                price_above_ema20 = bool(last.get('Close') > last.get('EMA20_T1'))
                price_below_ema20 = bool(last.get('Close') < last.get('EMA20_T1'))
                # BB genişlik oranı
                bb_width = None
                if not pd.isna(last.get('BBu_T1')) and not pd.isna(last.get('BBl_T1')) and last.get('BBu_T1') and last.get('BBl_T1'):
                    basis = (last['BBu_T1'] + last['BBl_T1']) / 2.0
                    if basis:
                        bb_width = (last['BBu_T1'] - last['BBl_T1']) / basis
                squeeze_ok = True if bb_width is None else (bb_width > 0.01)
                # Giriş koşulları
                long_cond = squeeze_ok and trend_long and vwap_long and mom_long and price_above_ema20
                short_cond = squeeze_ok and trend_short and vwap_short and mom_short and price_below_ema20
                if long_cond:
                    trend_text.append('LONG Confluence (5m)')
                if short_cond:
                    trend_text.append('SHORT Confluence (5m)')
                # Bilgi dökümü
                trend_text.append(f"EMA20/50: {'YUKARI' if last.get('EMA20_T1')>last.get('EMA50_T1') else 'AŞAĞI'} | VWAP: {'ÜSTÜ' if last.get('Close')>last.get('VWAP_T1') else 'ALTI'} | RSI9: {last.get('RSI9_T1'):.1f}")
                if trend_text:
                    self._append_info("Trend 1 (5dk): " + " | ".join(trend_text))
        except Exception:
            pass
        self._plot_signals(self.plot_frame1, self.latest_df, self.tf_var.get(), self.mum_var.get())
        self._plot_mum(self.plot_frame2, self.latest_df, self.tf_var.get(), self.mum_var.get(), highlight_signals=True)

    def _format_info(self, levels, info, df):
        txt = ""
        if levels:
            txt += f"Destek: {levels['support']:.4f}\n"
            txt += f"Direnç: {levels['resistance']:.4f}\n"
            txt += f"Alt Limit: {levels['limit_support']:.4f}\n"
            txt += f"Üst Limit: {levels['limit_resistance']:.4f}\n"
            txt += f"Yukarı Hedef: {levels['up_target']:.4f}\n"
            txt += f"Aşağı Hedef: {levels['down_target']:.4f}\n\n"
        if info:
            txt += f"Fiyat: {info.get('current_price', 'bilinmiyor')}\n"
            txt += f"Market Cap: {info.get('market_cap', 'bilinmiyor')}\n"
            txt += f"24s Hacim: {info.get('volume_24h', 'bilinmiyor')}\n"
        # add last signal info
        try:
            last = df.iloc[-1]
            sig = "—"
            if last.get('sig_buy_all'):
                sig = "BUY (ALL)"
            elif last.get('sig_sell_all'):
                sig = "SELL (ALL)"
            txt += f"\nSon Sinyal: {sig}\n"
        except Exception:
            pass
        return txt

    def _write_info(self, txt):
        self.info_txt.config(state="normal")
        self.info_txt.delete("1.0", "end")
        self.info_txt.insert("1.0", txt)
        self.info_txt.config(state="disabled")

    def _append_info(self, line):
        self.info_txt.config(state="normal")
        self.info_txt.insert("end", ("\n" if len(self.info_txt.get('1.0','end-1c'))>0 else "") + line)
        self.info_txt.config(state="disabled")

    def _plot_signals(self, frame, df, tf, bars):
        # safe-check
        if df is None or df.empty:
            for w in frame.winfo_children():
                w.destroy()
            lbl = ttk.Label(frame, text="Grafik için veri yok.")
            lbl.pack(fill="both", expand=True)
            return

        for w in frame.winfo_children():
            w.destroy()

        # prepare addplots (EMA lines, Bollinger, buy/sell markers)
        addplots = []

        # EMA lines
        if 'EMA50' in df.columns and 'EMA200' in df.columns:
            addplots.append(mpf.make_addplot(df['EMA50'], type='line', width=1))
            addplots.append(mpf.make_addplot(df['EMA200'], type='line', width=1))

        # Bollinger bands
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            addplots.append(mpf.make_addplot(df['BB_upper'], type='line', width=0.7))
            addplots.append(mpf.make_addplot(df['BB_mid'], type='line', width=0.7))
            addplots.append(mpf.make_addplot(df['BB_lower'], type='line', width=0.7))

        # Buy/sell markers (combined)
        try:
            buy_idx = df.index[df['sig_buy_all'] == True]
            sell_idx = df.index[df['sig_sell_all'] == True]
            if len(buy_idx):
                buy_y = df.loc[buy_idx]['Low'] * 0.995
                addplots.append(mpf.make_addplot(pd.Series(buy_y, index=buy_idx), type='scatter', markersize=80, marker='^'))
            if len(sell_idx):
                sell_y = df.loc[sell_idx]['High'] * 1.005
                addplots.append(mpf.make_addplot(pd.Series(sell_y, index=sell_idx), type='scatter', markersize=80, marker='v'))
        except Exception:
            pass

        # plot
        try:
            fig, axlist = mpf.plot(df, type='candle', style='yahoo', addplot=addplots, volume=True, show_nontrading=True, returnfig=True)
        except Exception as e:
            # if mplfinance fails, show a simple label
            for w in frame.winfo_children():
                w.destroy()
            lbl = ttk.Label(frame, text=f"Grafik çizilirken hata: {e}")
            lbl.pack(fill="both", expand=True)
            return

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas1 = canvas

    def _plot_mum(self, frame, df, tf, bars, highlight_signals=False):
        if df is None or df.empty:
            for w in frame.winfo_children():
                w.destroy()
            lbl = ttk.Label(frame, text="Grafik için veri yok.")
            lbl.pack(fill="both", expand=True)
            return

        for widget in frame.winfo_children():
            widget.destroy()

        addplots = []

        # If highlight_signals, show buy/sell markers
        if highlight_signals:
            try:
                buy_idx = df.index[df['sig_buy_all'] == True]
                sell_idx = df.index[df['sig_sell_all'] == True]
                if len(buy_idx):
                    buy_y = df.loc[buy_idx]['Low'] * 0.995
                    addplots.append(mpf.make_addplot(pd.Series(buy_y, index=buy_idx), type='scatter', markersize=80, marker='^'))
                if len(sell_idx):
                    sell_y = df.loc[sell_idx]['High'] * 1.005
                    addplots.append(mpf.make_addplot(pd.Series(sell_y, index=sell_idx), type='scatter', markersize=80, marker='v'))
            except Exception:
                pass

        # Also overlay EMA50/200 for this candle-only view
        if 'EMA50' in df.columns and 'EMA200' in df.columns:
            addplots.append(mpf.make_addplot(df['EMA50'], type='line', width=1))
            addplots.append(mpf.make_addplot(df['EMA200'], type='line', width=1))

        try:
            fig, axlist = mpf.plot(df, type='candle', style='yahoo', addplot=addplots, volume=True, show_nontrading=True, returnfig=True)
        except Exception as e:
            for w in frame.winfo_children():
                w.destroy()
            lbl = ttk.Label(frame, text=f"Grafik çizilirken hata: {e}")
            lbl.pack(fill="both", expand=True)
            return

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas2 = canvas

    def show_news(self):
        if not self.latest_news:
            messagebox.showinfo("Haberler", "Güncel haber bulunamadı.")
            return
        w = tk.Toplevel(self.root)
        w.title("Güncel Haberler")
        w.geometry("600x400")
        txt = tk.Text(w, wrap="word")
        txt.pack(fill="both", expand=True)
        for n in self.latest_news:
            txt.insert("end", f"{n['date']}\n{n['title']}\n{n['desc']}\n\n")
        txt.config(state="disabled")

def main():
    root = tk.Tk()
    app = CryptoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


            r.pack(side="left", padx=3)



        ms = ttk.Frame(top); ms.pack(side="left", padx=10)

        ttk.Label(ms, text="Mum:").pack(side="left")

        for m in MUM_OPTIONS:

            rb = ttk.Radiobutton(ms, text=f"{m} mum", value=m, variable=self.mum_var, command=self.on_change)

            rb.pack(side="left", padx=3)



        ctrl = ttk.Frame(top); ctrl.pack(side="right")

        self.refresh_btn = ttk.Button(ctrl, text="Yenile", command=self.update_now)

        self.refresh_btn.pack(side="left", padx=4)

        self.auto_btn = ttk.Button(ctrl, text="Otomatik Başlat", command=self.toggle_auto)

        self.auto_btn.pack(side="left", padx=4)



        main = ttk.Frame(self.root); main.pack(fill="both", expand=True, padx=8, pady=6)

        left_frame = ttk.Frame(main); left_frame.pack(side="left", fill="both", expand=True)

        right_frame = ttk.Frame(main, width=360); right_frame.pack(side="right", fill="y")



        # 1. Bölüm - Al-Sat Sinyalleri ve Grafik

        ttk.Label(left_frame, text="1. Bölüm: Al-Sat Sinyalleri ve Grafik", font=("Arial", 12, "bold")).pack(pady=6)

        self.plot_frame1 = ttk.Frame(left_frame)

        self.plot_frame1.pack(fill="both", expand=True)



        # 2. Bölüm - Sadece Mum Grafiği

        ttk.Label(left_frame, text="2. Bölüm: Sadece Mum Grafiği", font=("Arial", 12, "bold")).pack(pady=(20,6))

        self.plot_frame2 = ttk.Frame(left_frame)

        self.plot_frame2.pack(fill="both", expand=True)



        # Bilgiler ve Haberler

        ttk.Label(right_frame, text="Bilgiler & Haberler", font=("Arial", 12, "bold")).pack(pady=(6,4))

        self.info_txt = tk.Text(right_frame, width=44, height=30, wrap="word")

        self.info_txt.pack(padx=6, pady=4)

        self.info_txt.config(state="disabled")

        self.news_btn = ttk.Button(right_frame, text="Haberleri Göster", command=self.show_news)

        self.news_btn.pack(pady=6)



        self.status = ttk.Label(self.root, text="Hazır", relief="sunken", anchor="w")

        self.status.pack(side="bottom", fill="x")



    def set_status(self, txt):

        self.status.config(text=txt)



    def toggle_auto(self):

        if hasattr(self, "auto_thread") and self.auto_thread.is_alive():

            self.stop_flag.set()

            self.auto_btn.config(text="Otomatik Başlat")

            self.set_status("Otomatik durduruldu.")

        else:

            self.stop_flag.clear()

            self.auto_thread = threading.Thread(target=self._auto_loop, daemon=True)

            self.auto_thread.start()

            self.auto_btn.config(text="Otomatik Durdur")

            self.set_status("Otomatik başlatıldı.")



    def _auto_loop(self):

        while not self.stop_flag.is_set():

            self.update_now()

            for _ in range(REFRESH_SECONDS):

                if self.stop_flag.is_set():

                    break

                time.sleep(1)



    def on_change(self):

        self.update_now()



    def update_now(self):

        t = threading.Thread(target=self._worker, daemon=True)

        t.start()



    def _worker(self):

        coin_label = self.coin_var.get()

        mapping = {c[0]:c for c in COINS}

        if coin_label not in mapping:

            self.set_status("Coin bulunamadı.")

            return

        label, yf_ticker, cg_id, ku_sym, kr_sym, cb_sym = mapping[coin_label]

        tf = self.tf_var.get()

        bars = self.mum_var.get()

        self.set_status(f"Veri çekiliyor: {label} | {tf} dk | {bars} mum ...")



        df = None

        source = None

        try_order = [

            ("CoinGecko", lambda: fetch_coingecko(cg_id, tf, bars)),

            ("KuCoin", lambda: fetch_kucoin(ku_sym, tf, bars)),

            ("Kraken", lambda: fetch_kraken(kr_sym, tf, bars)),

            ("Coinbase", lambda: fetch_coinbase(cb_sym, tf, bars)),

            ("Yahoo", lambda: fetch_yfinance(yf_ticker, tf, bars))

        ]

        for name, func in try_order:

            try:

                df = func()

                if df is not None and not df.empty:

                    source = name

                    break

            except Exception:

                df = None

        if df is None or df.empty:

            self.set_status("Veri alınamadı. İnternet/VPN/Firewall kontrol et.")

            self._write_info("Veri alınamadı. Lütfen bağlantınızı kontrol edin.")

            return



        # ensure datetime index and proper columns

        df.index = pd.to_datetime(df.index)

        df = df[["Open","High","Low","Close","Volume"]].copy()



        # ---------- ADD: Indicators calculation ----------

        # RSI

        df['RSI'] = calculate_rsi(df, period=14)



        # MACD

        macd, macd_signal, macd_hist = calculate_macd(df, 12, 26, 9)

        df['MACD'] = macd

        df['MACD_signal'] = macd_signal

        df['MACD_hist'] = macd_hist



        # Bollinger Bands

        bb_upper, bb_lower = calculate_bollinger_bands(df, window=20, num_std=2)

        df['BB_upper'] = bb_upper

        df['BB_lower'] = bb_lower

        df['BB_mid'] = (bb_upper + bb_lower) / 2



        # EMAs

        df['EMA50'] = calculate_ema(df, 50)

        df['EMA200'] = calculate_ema(df, 200)



        # Stochastic

        stoch_k, stoch_d = calculate_stochastic(df, k_window=14, d_window=3)

        df['STOCH_K'] = stoch_k

        df['STOCH_D'] = stoch_d



        # Generate individual signals

        df['sig_rsi_buy'] = df['RSI'] < 30

        df['sig_rsi_sell'] = df['RSI'] > 70

        df['sig_macd_buy'] = (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)

        df['sig_macd_sell'] = (df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) >= 0)

        df['sig_bb_buy'] = (df['Close'] <= df['BB_lower'] * 1.002)  # close near lower band

        df['sig_bb_sell'] = (df['Close'] >= df['BB_upper'] * 0.998)  # close near upper band

        df['sig_ema_bull'] = df['EMA50'] > df['EMA200']

        df['sig_ema_bear'] = df['EMA50'] < df['EMA200']

        df['sig_stoch_buy'] = (df['STOCH_K'] < 20) & (df['STOCH_K'].shift(1) < df['STOCH_D'].shift(1)) & (df['STOCH_K'] > df['STOCH_D'])

        df['sig_stoch_sell'] = (df['STOCH_K'] > 80) & (df['STOCH_K'].shift(1) > df['STOCH_D'].shift(1)) & (df['STOCH_K'] < df['STOCH_D'])



        # Combined signals: require most indicators to agree

        df['sig_buy_all'] = (

            df['sig_rsi_buy'] &

            df['sig_macd_buy'].fillna(False) &

            df['sig_bb_buy'].fillna(False) &

            df['sig_ema_bull'].fillna(False) &

            df['sig_stoch_buy'].fillna(False)

        )

        df['sig_sell_all'] = (

            df['sig_rsi_sell'] &

            df['sig_macd_sell'].fillna(False) &

            df['sig_bb_sell'].fillna(False) &

            df['sig_ema_bear'].fillna(False) &

            df['sig_stoch_sell'].fillna(False)

        )

        # ---------- END: Indicators ----------



        levels = calc_levels(df, bars)

        info = get_market_info(cg_id)

        news = get_news(cg_id)



        self.latest_df = df

        self.latest_levels = levels

        self.latest_info = info

        self.latest_news = news



        self.root.after(0, self._update_gui)

        self.set_status(f"Veri güncellendi ({source}).")



    def _update_gui(self):

        # If no data, skip

        if self.latest_df is None or self.latest_df.empty:

            self._write_info("Grafik için yeterli veri yok.")

            return

        self._write_info(self._format_info(self.latest_levels, self.latest_info, self.latest_df))

        self._plot_signals(self.plot_frame1, self.latest_df, self.tf_var.get(), self.mum_var.get())

        self._plot_mum(self.plot_frame2, self.latest_df, self.tf_var.get(), self.mum_var.get(), highlight_signals=True)



    def _format_info(self, levels, info, df):

        txt = ""

        if levels:

            txt += f"Destek: {levels['support']:.4f}\n"

            txt += f"Direnç: {levels['resistance']:.4f}\n"

            txt += f"Alt Limit: {levels['limit_support']:.4f}\n"

            txt += f"Üst Limit: {levels['limit_resistance']:.4f}\n"

            txt += f"Yukarı Hedef: {levels['up_target']:.4f}\n"

            txt += f"Aşağı Hedef: {levels['down_target']:.4f}\n\n"

        if info:

            txt += f"Fiyat: {info.get('current_price', 'bilinmiyor')}\n"

            txt += f"Market Cap: {info.get('market_cap', 'bilinmiyor')}\n"

            txt += f"24s Hacim: {info.get('volume_24h', 'bilinmiyor')}\n"

        # add last signal info

        try:

            last = df.iloc[-1]

            sig = "—"

            if last.get('sig_buy_all'):

                sig = "BUY (ALL)"

            elif last.get('sig_sell_all'):

                sig = "SELL (ALL)"

            txt += f"\nSon Sinyal: {sig}\n"

        except Exception:

            pass

        return txt



    def _write_info(self, txt):

        self.info_txt.config(state="normal")

        self.info_txt.delete("1.0", "end")

        self.info_txt.insert("1.0", txt)

        self.info_txt.config(state="disabled")



    def _plot_signals(self, frame, df, tf, bars):

        # safe-check

        if df is None or df.empty:

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text="Grafik için veri yok.")

            lbl.pack(fill="both", expand=True)

            return



        for w in frame.winfo_children():

            w.destroy()



        # prepare addplots (EMA lines, Bollinger, buy/sell markers)

        addplots = []



        # EMA lines

        if 'EMA50' in df.columns and 'EMA200' in df.columns:

            addplots.append(mpf.make_addplot(df['EMA50'], type='line', width=1))

            addplots.append(mpf.make_addplot(df['EMA200'], type='line', width=1))



        # Bollinger bands

        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:

            addplots.append(mpf.make_addplot(df['BB_upper'], type='line', width=0.7))

            addplots.append(mpf.make_addplot(df['BB_mid'], type='line', width=0.7))

            addplots.append(mpf.make_addplot(df['BB_lower'], type='line', width=0.7))



        # Buy/sell markers (combined)

        try:

            buy_idx = df.index[df['sig_buy_all'] == True]

            sell_idx = df.index[df['sig_sell_all'] == True]

            if len(buy_idx):

                buy_y = df.loc[buy_idx]['Low'] * 0.995

                addplots.append(mpf.make_addplot(pd.Series(buy_y, index=buy_idx), type='scatter', markersize=80, marker='^'))

            if len(sell_idx):

                sell_y = df.loc[sell_idx]['High'] * 1.005

                addplots.append(mpf.make_addplot(pd.Series(sell_y, index=sell_idx), type='scatter', markersize=80, marker='v'))

        except Exception:

            pass



        # plot

        try:

            fig, axlist = mpf.plot(df, type='candle', style='yahoo', addplot=addplots, volume=True, show_nontrading=True, returnfig=True)

        except Exception as e:

            # if mplfinance fails, show a simple label

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text=f"Grafik çizilirken hata: {e}")

            lbl.pack(fill="both", expand=True)

            return



        canvas = FigureCanvasTkAgg(fig, master=frame)

        canvas.draw()

        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas1 = canvas



    def _plot_mum(self, frame, df, tf, bars, highlight_signals=False):

        if df is None or df.empty:

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text="Grafik için veri yok.")

            lbl.pack(fill="both", expand=True)

            return



        for widget in frame.winfo_children():

            widget.destroy()



        addplots = []



        # If highlight_signals, show buy/sell markers

        if highlight_signals:

            try:

                buy_idx = df.index[df['sig_buy_all'] == True]

                sell_idx = df.index[df['sig_sell_all'] == True]

                if len(buy_idx):

                    buy_y = df.loc[buy_idx]['Low'] * 0.995

                    addplots.append(mpf.make_addplot(pd.Series(buy_y, index=buy_idx), type='scatter', markersize=80, marker='^'))

                if len(sell_idx):

                    sell_y = df.loc[sell_idx]['High'] * 1.005

                    addplots.append(mpf.make_addplot(pd.Series(sell_y, index=sell_idx), type='scatter', markersize=80, marker='v'))

            except Exception:

                pass



        # Also overlay EMA50/200 for this candle-only view

        if 'EMA50' in df.columns and 'EMA200' in df.columns:

            addplots.append(mpf.make_addplot(df['EMA50'], type='line', width=1))

            addplots.append(mpf.make_addplot(df['EMA200'], type='line', width=1))



        try:

            fig, axlist = mpf.plot(df, type='candle', style='yahoo', addplot=addplots, volume=True, show_nontrading=True, returnfig=True)

        except Exception as e:

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text=f"Grafik çizilirken hata: {e}")

            lbl.pack(fill="both", expand=True)

            return



        canvas = FigureCanvasTkAgg(fig, master=frame)

        canvas.draw()

        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas2 = canvas



    def show_news(self):

        if not self.latest_news:

            messagebox.showinfo("Haberler", "Güncel haber bulunamadı.")

            return

        w = tk.Toplevel(self.root)

        w.title("Güncel Haberler")

        w.geometry("600x400")

        txt = tk.Text(w, wrap="word")

        txt.pack(fill="both", expand=True)

        for n in self.latest_news:

            txt.insert("end", f"{n['date']}\n{n['title']}\n{n['desc']}\n\n")

        txt.config(state="disabled")



def main():

    root = tk.Tk()

    app = CryptoApp(root)

    root.mainloop()



if __name__ == "__main__":

    main()




            r.pack(side="left", padx=3)



        ms = ttk.Frame(top); ms.pack(side="left", padx=10)

        ttk.Label(ms, text="Mum:").pack(side="left")

        for m in MUM_OPTIONS:

            rb = ttk.Radiobutton(ms, text=f"{m} mum", value=m, variable=self.mum_var, command=self.on_change)

            rb.pack(side="left", padx=3)



        ctrl = ttk.Frame(top); ctrl.pack(side="right")

        self.refresh_btn = ttk.Button(ctrl, text="Yenile", command=self.update_now)

        self.refresh_btn.pack(side="left", padx=4)

        self.auto_btn = ttk.Button(ctrl, text="Otomatik Başlat", command=self.toggle_auto)

        self.auto_btn.pack(side="left", padx=4)



        main = ttk.Frame(self.root); main.pack(fill="both", expand=True, padx=8, pady=6)

        left_frame = ttk.Frame(main); left_frame.pack(side="left", fill="both", expand=True)

        right_frame = ttk.Frame(main, width=360); right_frame.pack(side="right", fill="y")



        # 1. Bölüm - Al-Sat Sinyalleri ve Grafik

        ttk.Label(left_frame, text="1. Bölüm: Al-Sat Sinyalleri ve Grafik", font=("Arial", 12, "bold")).pack(pady=6)

        self.plot_frame1 = ttk.Frame(left_frame)

        self.plot_frame1.pack(fill="both", expand=True)



        # 2. Bölüm - Sadece Mum Grafiği

        ttk.Label(left_frame, text="2. Bölüm: Sadece Mum Grafiği", font=("Arial", 12, "bold")).pack(pady=(20,6))

        self.plot_frame2 = ttk.Frame(left_frame)

        self.plot_frame2.pack(fill="both", expand=True)



        # Bilgiler ve Haberler

        ttk.Label(right_frame, text="Bilgiler & Haberler", font=("Arial", 12, "bold")).pack(pady=(6,4))

        self.info_txt = tk.Text(right_frame, width=44, height=30, wrap="word")

        self.info_txt.pack(padx=6, pady=4)

        self.info_txt.config(state="disabled")

        self.news_btn = ttk.Button(right_frame, text="Haberleri Göster", command=self.show_news)

        self.news_btn.pack(pady=6)



        self.status = ttk.Label(self.root, text="Hazır", relief="sunken", anchor="w")

        self.status.pack(side="bottom", fill="x")



    def set_status(self, txt):

        self.status.config(text=txt)



    def toggle_auto(self):

        if hasattr(self, "auto_thread") and self.auto_thread.is_alive():

            self.stop_flag.set()

            self.auto_btn.config(text="Otomatik Başlat")

            self.set_status("Otomatik durduruldu.")

        else:

            self.stop_flag.clear()

            self.auto_thread = threading.Thread(target=self._auto_loop, daemon=True)

            self.auto_thread.start()

            self.auto_btn.config(text="Otomatik Durdur")

            self.set_status("Otomatik başlatıldı.")



    def _auto_loop(self):

        while not self.stop_flag.is_set():

            self.update_now()

            for _ in range(REFRESH_SECONDS):

                if self.stop_flag.is_set():

                    break

                time.sleep(1)



    def on_change(self):

        self.update_now()



    def update_now(self):

        t = threading.Thread(target=self._worker, daemon=True)

        t.start()



    def _worker(self):

        coin_label = self.coin_var.get()

        mapping = {c[0]:c for c in COINS}

        if coin_label not in mapping:

            self.set_status("Coin bulunamadı.")

            return

        label, yf_ticker, cg_id, ku_sym, kr_sym, cb_sym = mapping[coin_label]

        tf = self.tf_var.get()

        bars = self.mum_var.get()

        self.set_status(f"Veri çekiliyor: {label} | {tf} dk | {bars} mum ...")



        df = None

        source = None

        try_order = [

            ("CoinGecko", lambda: fetch_coingecko(cg_id, tf, bars)),

            ("KuCoin", lambda: fetch_kucoin(ku_sym, tf, bars)),

            ("Kraken", lambda: fetch_kraken(kr_sym, tf, bars)),

            ("Coinbase", lambda: fetch_coinbase(cb_sym, tf, bars)),

            ("Yahoo", lambda: fetch_yfinance(yf_ticker, tf, bars))

        ]

        for name, func in try_order:

            try:

                df = func()

                if df is not None and not df.empty:

                    source = name

                    break

            except Exception:

                df = None

        if df is None or df.empty:

            self.set_status("Veri alınamadı. İnternet/VPN/Firewall kontrol et.")

            self._write_info("Veri alınamadı. Lütfen bağlantınızı kontrol edin.")

            return



        # ensure datetime index and proper columns

        df.index = pd.to_datetime(df.index)

        df = df[["Open","High","Low","Close","Volume"]].copy()



        # ---------- ADD: Indicators calculation ----------

        # RSI

        df['RSI'] = calculate_rsi(df, period=14)



        # MACD

        macd, macd_signal, macd_hist = calculate_macd(df, 12, 26, 9)

        df['MACD'] = macd

        df['MACD_signal'] = macd_signal

        df['MACD_hist'] = macd_hist



        # Bollinger Bands

        bb_upper, bb_lower = calculate_bollinger_bands(df, window=20, num_std=2)

        df['BB_upper'] = bb_upper

        df['BB_lower'] = bb_lower

        df['BB_mid'] = (bb_upper + bb_lower) / 2



        # EMAs

        df['EMA50'] = calculate_ema(df, 50)

        df['EMA200'] = calculate_ema(df, 200)



        # Stochastic

        stoch_k, stoch_d = calculate_stochastic(df, k_window=14, d_window=3)

        df['STOCH_K'] = stoch_k

        df['STOCH_D'] = stoch_d



        # Generate individual signals

        df['sig_rsi_buy'] = df['RSI'] < 30

        df['sig_rsi_sell'] = df['RSI'] > 70

        df['sig_macd_buy'] = (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)

        df['sig_macd_sell'] = (df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) >= 0)

        df['sig_bb_buy'] = (df['Close'] <= df['BB_lower'] * 1.002)  # close near lower band

        df['sig_bb_sell'] = (df['Close'] >= df['BB_upper'] * 0.998)  # close near upper band

        df['sig_ema_bull'] = df['EMA50'] > df['EMA200']

        df['sig_ema_bear'] = df['EMA50'] < df['EMA200']

        df['sig_stoch_buy'] = (df['STOCH_K'] < 20) & (df['STOCH_K'].shift(1) < df['STOCH_D'].shift(1)) & (df['STOCH_K'] > df['STOCH_D'])

        df['sig_stoch_sell'] = (df['STOCH_K'] > 80) & (df['STOCH_K'].shift(1) > df['STOCH_D'].shift(1)) & (df['STOCH_K'] < df['STOCH_D'])



        # Combined signals: require most indicators to agree

        df['sig_buy_all'] = (

            df['sig_rsi_buy'] &

            df['sig_macd_buy'].fillna(False) &

            df['sig_bb_buy'].fillna(False) &

            df['sig_ema_bull'].fillna(False) &

            df['sig_stoch_buy'].fillna(False)

        )

        df['sig_sell_all'] = (

            df['sig_rsi_sell'] &

            df['sig_macd_sell'].fillna(False) &

            df['sig_bb_sell'].fillna(False) &

            df['sig_ema_bear'].fillna(False) &

            df['sig_stoch_sell'].fillna(False)

        )

        # ---------- END: Indicators ----------



        levels = calc_levels(df, bars)

        info = get_market_info(cg_id)

        news = get_news(cg_id)



        self.latest_df = df

        self.latest_levels = levels

        self.latest_info = info

        self.latest_news = news



        self.root.after(0, self._update_gui)

        self.set_status(f"Veri güncellendi ({source}).")



    def _update_gui(self):

        # If no data, skip

        if self.latest_df is None or self.latest_df.empty:

            self._write_info("Grafik için yeterli veri yok.")

            return

        self._write_info(self._format_info(self.latest_levels, self.latest_info, self.latest_df))

        self._plot_signals(self.plot_frame1, self.latest_df, self.tf_var.get(), self.mum_var.get())

        self._plot_mum(self.plot_frame2, self.latest_df, self.tf_var.get(), self.mum_var.get(), highlight_signals=True)



    def _format_info(self, levels, info, df):

        txt = ""

        if levels:

            txt += f"Destek: {levels['support']:.4f}\n"

            txt += f"Direnç: {levels['resistance']:.4f}\n"

            txt += f"Alt Limit: {levels['limit_support']:.4f}\n"

            txt += f"Üst Limit: {levels['limit_resistance']:.4f}\n"

            txt += f"Yukarı Hedef: {levels['up_target']:.4f}\n"

            txt += f"Aşağı Hedef: {levels['down_target']:.4f}\n\n"

        if info:

            txt += f"Fiyat: {info.get('current_price', 'bilinmiyor')}\n"

            txt += f"Market Cap: {info.get('market_cap', 'bilinmiyor')}\n"

            txt += f"24s Hacim: {info.get('volume_24h', 'bilinmiyor')}\n"

        # add last signal info

        try:

            last = df.iloc[-1]

            sig = "—"

            if last.get('sig_buy_all'):

                sig = "BUY (ALL)"

            elif last.get('sig_sell_all'):

                sig = "SELL (ALL)"

            txt += f"\nSon Sinyal: {sig}\n"

        except Exception:

            pass

        return txt



    def _write_info(self, txt):

        self.info_txt.config(state="normal")

        self.info_txt.delete("1.0", "end")

        self.info_txt.insert("1.0", txt)

        self.info_txt.config(state="disabled")



    def _plot_signals(self, frame, df, tf, bars):

        # safe-check

        if df is None or df.empty:

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text="Grafik için veri yok.")

            lbl.pack(fill="both", expand=True)

            return



        for w in frame.winfo_children():

            w.destroy()



        # prepare addplots (EMA lines, Bollinger, buy/sell markers)

        addplots = []



        # EMA lines

        if 'EMA50' in df.columns and 'EMA200' in df.columns:

            addplots.append(mpf.make_addplot(df['EMA50'], type='line', width=1))

            addplots.append(mpf.make_addplot(df['EMA200'], type='line', width=1))



        # Bollinger bands

        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:

            addplots.append(mpf.make_addplot(df['BB_upper'], type='line', width=0.7))

            addplots.append(mpf.make_addplot(df['BB_mid'], type='line', width=0.7))

            addplots.append(mpf.make_addplot(df['BB_lower'], type='line', width=0.7))



        # Buy/sell markers (combined)

        try:

            buy_idx = df.index[df['sig_buy_all'] == True]

            sell_idx = df.index[df['sig_sell_all'] == True]

            if len(buy_idx):

                buy_y = df.loc[buy_idx]['Low'] * 0.995

                addplots.append(mpf.make_addplot(pd.Series(buy_y, index=buy_idx), type='scatter', markersize=80, marker='^'))

            if len(sell_idx):

                sell_y = df.loc[sell_idx]['High'] * 1.005

                addplots.append(mpf.make_addplot(pd.Series(sell_y, index=sell_idx), type='scatter', markersize=80, marker='v'))

        except Exception:

            pass



        # plot

        try:

            fig, axlist = mpf.plot(df, type='candle', style='yahoo', addplot=addplots, volume=True, show_nontrading=True, returnfig=True)

        except Exception as e:

            # if mplfinance fails, show a simple label

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text=f"Grafik çizilirken hata: {e}")

            lbl.pack(fill="both", expand=True)

            return



        canvas = FigureCanvasTkAgg(fig, master=frame)

        canvas.draw()

        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas1 = canvas



    def _plot_mum(self, frame, df, tf, bars, highlight_signals=False):

        if df is None or df.empty:

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text="Grafik için veri yok.")

            lbl.pack(fill="both", expand=True)

            return



        for widget in frame.winfo_children():

            widget.destroy()



        addplots = []



        # If highlight_signals, show buy/sell markers

        if highlight_signals:

            try:

                buy_idx = df.index[df['sig_buy_all'] == True]

                sell_idx = df.index[df['sig_sell_all'] == True]

                if len(buy_idx):

                    buy_y = df.loc[buy_idx]['Low'] * 0.995

                    addplots.append(mpf.make_addplot(pd.Series(buy_y, index=buy_idx), type='scatter', markersize=80, marker='^'))

                if len(sell_idx):

                    sell_y = df.loc[sell_idx]['High'] * 1.005

                    addplots.append(mpf.make_addplot(pd.Series(sell_y, index=sell_idx), type='scatter', markersize=80, marker='v'))

            except Exception:

                pass



        # Also overlay EMA50/200 for this candle-only view

        if 'EMA50' in df.columns and 'EMA200' in df.columns:

            addplots.append(mpf.make_addplot(df['EMA50'], type='line', width=1))

            addplots.append(mpf.make_addplot(df['EMA200'], type='line', width=1))



        try:

            fig, axlist = mpf.plot(df, type='candle', style='yahoo', addplot=addplots, volume=True, show_nontrading=True, returnfig=True)

        except Exception as e:

            for w in frame.winfo_children():

                w.destroy()

            lbl = ttk.Label(frame, text=f"Grafik çizilirken hata: {e}")

            lbl.pack(fill="both", expand=True)

            return



        canvas = FigureCanvasTkAgg(fig, master=frame)

        canvas.draw()

        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas2 = canvas



    def show_news(self):

        if not self.latest_news:

            messagebox.showinfo("Haberler", "Güncel haber bulunamadı.")

            return

        w = tk.Toplevel(self.root)

        w.title("Güncel Haberler")

        w.geometry("600x400")

        txt = tk.Text(w, wrap="word")

        txt.pack(fill="both", expand=True)

        for n in self.latest_news:

            txt.insert("end", f"{n['date']}\n{n['title']}\n{n['desc']}\n\n")

        txt.config(state="disabled")



def main():

    root = tk.Tk()

    app = CryptoApp(root)

    root.mainloop()



if __name__ == "__main__":

    main()


