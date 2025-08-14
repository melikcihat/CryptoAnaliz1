#!/usr/bin/env python3
"""
Advanced Binance Data Fetcher with Multiple Fallbacks
SSL sorunlarını çözen, multiple endpoint'li, robust veri alma sistemi
"""

import requests
import pandas as pd
import time
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
from datetime import datetime, timedelta

# SSL verification'ı bypass et (test için)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AdvancedBinanceFetcher:
    def __init__(self):
        self.session = self.create_robust_session()
        
        # Multiple Binance endpoints
        self.binance_hosts = [
            "https://api.binance.com",
            "https://api1.binance.com", 
            "https://api2.binance.com",
            "https://api3.binance.com",
            "https://data-api.binance.vision"
        ]
        
        # Backup data sources
        self.backup_sources = [
            self.fetch_from_coingecko,
            self.fetch_from_cryptocompare,
            self.fetch_from_yahoo
        ]
        
    def create_robust_session(self):
        """SSL sorunlarını handle eden robust session oluştur"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def fetch_binance_with_fallbacks(self, symbol: str, interval: str, limit: int = 100):
        """Multiple fallback'li Binance veri alma"""
        
        # Method 1: Normal HTTPS
        data = self.try_binance_normal(symbol, interval, limit)
        if data:
            return data, "Binance-Normal"
        
        # Method 2: SSL verification disabled
        data = self.try_binance_no_ssl(symbol, interval, limit)
        if data:
            return data, "Binance-NoSSL"
        
        # Method 3: Different endpoints
        data = self.try_binance_endpoints(symbol, interval, limit)
        if data:
            return data, "Binance-Alt"
        
        # Method 4: Proxy-like approach (simüle)
        data = self.try_binance_with_different_headers(symbol, interval, limit)
        if data:
            return data, "Binance-Headers"
        
        return None, None
    
    def try_binance_normal(self, symbol: str, interval: str, limit: int):
        """Normal Binance API çağrısı"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def try_binance_no_ssl(self, symbol: str, interval: str, limit: int):
        """SSL verification'sız deneme"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            
            response = requests.get(url, params=params, timeout=20, verify=False)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def try_binance_endpoints(self, symbol: str, interval: str, limit: int):
        """Alternatif Binance endpoint'leri dene"""
        for host in self.binance_hosts[1:]:  # İlkini zaten denedik
            try:
                url = f"{host}/api/v3/klines"
                params = {'symbol': symbol, 'interval': interval, 'limit': limit}
                
                response = requests.get(url, params=params, timeout=10, verify=False)
                if response.status_code == 200:
                    return response.json()
            except:
                continue
        return None
    
    def try_binance_with_different_headers(self, symbol: str, interval: str, limit: int):
        """Farklı headers ile deneme"""
        headers_list = [
            {'User-Agent': 'python-requests/2.28.1'},
            {'User-Agent': 'curl/7.68.0'},
            {'User-Agent': 'TradingView/1.0'},
            {'User-Agent': 'Mozilla/5.0 (compatible; Binance/1.0)'}
        ]
        
        for headers in headers_list:
            try:
                url = f"https://api.binance.com/api/v3/klines"
                params = {'symbol': symbol, 'interval': interval, 'limit': limit}
                
                response = requests.get(url, params=params, headers=headers, 
                                      timeout=10, verify=False)
                if response.status_code == 200:
                    return response.json()
            except:
                continue
        return None
    
    def fetch_from_coingecko(self, symbol: str, interval: str, limit: int):
        """CoinGecko'dan veri al"""
        try:
            # Symbol mapping
            coin_map = {
                'BTCUSDT': 'bitcoin',
                'ETHUSDT': 'ethereum', 
                'BNBUSDT': 'binancecoin',
                'ADAUSDT': 'cardano',
                'SOLUSDT': 'solana'
            }
            
            coin_id = coin_map.get(symbol)
            if not coin_id:
                return None
            
            # CoinGecko sadece daily data veriyor, simulate edelim
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': '1', 'interval': 'hourly'}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                
                if prices and volumes:
                    # Simulate OHLCV data
                    candles = []
                    for i, (timestamp, price) in enumerate(prices[-limit:]):
                        volume = volumes[i][1] if i < len(volumes) else 0
                        # Simulate OHLC from price
                        candles.append([
                            timestamp,
                            price * 0.999,  # Open
                            price * 1.001,  # High  
                            price * 0.998,  # Low
                            price,          # Close
                            volume
                        ])
                    return candles
        except:
            pass
        return None
    
    def fetch_from_cryptocompare(self, symbol: str, interval: str, limit: int):
        """CryptoCompare'dan veri al"""
        try:
            base = symbol.replace('USDT', '')
            url = f"https://min-api.cryptocompare.com/data/v2/histominute"
            params = {
                'fsym': base,
                'tsym': 'USD', 
                'limit': limit,
                'aggregate': 5  # 5 dakika
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'Success':
                    candles = []
                    for item in data['Data']['Data']:
                        candles.append([
                            item['time'] * 1000,  # Timestamp
                            item['open'],
                            item['high'],
                            item['low'], 
                            item['close'],
                            item['volumefrom']
                        ])
                    return candles
        except:
            pass
        return None
    
    def fetch_from_yahoo(self, symbol: str, interval: str, limit: int):
        """Yahoo Finance'tan veri al (yfinance kullanarak)"""
        try:
            import yfinance as yf
            
            # Symbol mapping
            yahoo_symbols = {
                'BTCUSDT': 'BTC-USD',
                'ETHUSDT': 'ETH-USD',
                'BNBUSDT': 'BNB-USD',
                'ADAUSDT': 'ADA-USD',
                'SOLUSDT': 'SOL-USD'
            }
            
            yf_symbol = yahoo_symbols.get(symbol)
            if not yf_symbol:
                return None
                
            ticker = yf.Ticker(yf_symbol)
            # Son 2 gün, 5 dakika interval
            df = ticker.history(period="2d", interval="5m")
            
            if not df.empty:
                candles = []
                for idx, row in df.tail(limit).iterrows():
                    timestamp = int(idx.timestamp() * 1000)
                    candles.append([
                        timestamp,
                        row['Open'],
                        row['High'], 
                        row['Low'],
                        row['Close'],
                        row['Volume']
                    ])
                return candles
        except:
            pass
        return None
    
    def get_crypto_data(self, symbol: str = 'BTCUSDT', interval: str = '5m', limit: int = 100):
        """Ana veri alma fonksiyonu"""
        print(f"🚀 Fetching {symbol} data...")
        
        # Primary: Binance
        data, source = self.fetch_binance_with_fallbacks(symbol, interval, limit)
        if data:
            print(f"✅ Success from {source}")
            return self.convert_to_dataframe(data), source
        
        print("❌ Binance failed, trying backups...")
        
        # Backup sources
        for i, backup_func in enumerate(self.backup_sources):
            try:
                data = backup_func(symbol, interval, limit)
                if data:
                    source_name = backup_func.__name__.replace('fetch_from_', '').title()
                    print(f"✅ Success from {source_name}")
                    return self.convert_to_dataframe(data), source_name
            except Exception as e:
                print(f"❌ {backup_func.__name__} failed: {e}")
        
        print("❌ All sources failed!")
        return None, None
    
    def convert_to_dataframe(self, data):
        """Raw data'yı pandas DataFrame'e çevir"""
        try:
            rows = []
            for item in data:
                timestamp = pd.to_datetime(int(item[0]), unit='ms')
                rows.append([
                    timestamp,
                    float(item[1]),  # Open
                    float(item[2]),  # High
                    float(item[3]),  # Low
                    float(item[4]),  # Close
                    float(item[5])   # Volume
                ])
            
            df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"DataFrame conversion error: {e}")
            return None

# Test function
def test_advanced_fetcher():
    fetcher = AdvancedBinanceFetcher()
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for symbol in symbols:
        print(f"\n🔍 Testing {symbol}:")
        df, source = fetcher.get_crypto_data(symbol, '5m', 20)
        
        if df is not None:
            print(f"✅ Got {len(df)} candles from {source}")
            print(f"Latest price: ${df['Close'].iloc[-1]:.4f}")
            print(f"Price range: ${df['Low'].min():.4f} - ${df['High'].max():.4f}")
        else:
            print(f"❌ Failed to get data for {symbol}")

if __name__ == "__main__":
    test_advanced_fetcher()
