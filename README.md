# 🚀 Crypto Trading Terminal

Professional TradingView-style cryptocurrency analysis platform with real-time data and technical indicators.

## ✨ Features

- **📊 Professional Interface**: TradingView-inspired design
- **🔄 Multi-Source Data**: Yahoo Finance, Binance, KuCoin, Coinbase, Kraken, Bitstamp
- **📈 Technical Analysis**: EMA, VWAP, RSI, MACD, Bollinger Bands
- **🎯 Trading Signals**: Buy/Sell signals with confluence analysis
- **⏰ Multiple Timeframes**: 1m, 3m, 5m, 15m, 30m, 45m, 1h, 3h
- **📅 Flexible Ranges**: 6h, 24h, 3d, 1m, 3m, 6m, 1y, All
- **🌐 Real-time Updates**: Live price and volume data

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Charts**: Plotly
- **Data**: Pandas, NumPy
- **APIs**: yfinance, requests
- **Real-time**: WebSocket (Binance)

## 🚀 Live Demo

[View Live App](https://your-app-url.streamlit.app)

## 🔧 Local Setup

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📋 Supported Assets

- BTC, ETH, BNB, ADA, SOL
- XRP, DOGE, LTC, MATIC, DOT

## 📈 Technical Indicators

### Trend 1 Strategy
- **EMA20 & EMA50**: Trend filters
- **VWAP**: Intraday balance
- **RSI(9)**: Momentum
- **MACD(8,21,5)**: Momentum confirmation  
- **Bollinger Bands(20,2)**: Volatility/breakout

## 🎯 Trading Signals

Confluence-based buy/sell signals with:
- Trend alignment (EMA20 > EMA50)
- VWAP position
- Momentum confirmation (RSI + MACD)
- Price above/below EMA20

## 📊 Data Sources Priority

1. **Yahoo Finance** (Primary - stable)
2. **Bitstamp** (Reliable EU exchange)
3. **KuCoin, Coinbase, Kraken** (Backup)
4. **Binance** (High-frequency, may have regional restrictions)
5. **Demo Data** (Fallback simulation)

---

*Built with ❤️ for crypto traders*
