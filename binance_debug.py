#!/usr/bin/env python3
"""
Binance API Debug Script
Binance veri alma sorunlarını detaylı tespit eder
"""

import requests
import json
import time
from datetime import datetime, timedelta

def test_basic_connection():
    """Temel bağlantıyı test et"""
    print("🔍 Temel Bağlantı Testi...")
    
    hosts = [
        "https://api.binance.com",
        "https://api1.binance.com", 
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://data-api.binance.vision"
    ]
    
    for host in hosts:
        try:
            url = f"{host}/api/v3/ping"
            response = requests.get(url, timeout=5)
            print(f"✅ {host}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ {host}: {str(e)}")
    print()

def test_server_time():
    """Server time test"""
    print("⏰ Server Time Testi...")
    try:
        url = "https://api.binance.com/api/v3/time"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            server_time = datetime.fromtimestamp(data['serverTime'] / 1000)
            local_time = datetime.now()
            diff = abs((server_time - local_time).total_seconds())
            print(f"✅ Server Time: {server_time}")
            print(f"✅ Local Time: {local_time}")
            print(f"✅ Difference: {diff:.2f} seconds")
        else:
            print(f"❌ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

def test_symbol_info():
    """Symbol bilgilerini test et"""
    print("💰 Symbol Info Testi...")
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            symbols = [s['symbol'] for s in data['symbols'] if s['symbol'].endswith('USDT')]
            crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            
            print(f"✅ Total USDT pairs: {len(symbols)}")
            for symbol in crypto_symbols:
                if symbol in symbols:
                    print(f"✅ {symbol}: Available")
                else:
                    print(f"❌ {symbol}: Not found")
        else:
            print(f"❌ Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

def test_klines_data():
    """Kline verilerini test et"""
    print("📊 Klines Data Testi...")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '3h']
    
    for symbol in symbols:
        print(f"\n🪙 Testing {symbol}:")
        for interval in intervals:
            try:
                url = f"https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': 10
                }
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        first_kline = data[0]
                        open_time = datetime.fromtimestamp(int(first_kline[0]) / 1000)
                        open_price = float(first_kline[1])
                        close_price = float(first_kline[4])
                        volume = float(first_kline[5])
                        print(f"  ✅ {interval}: {len(data)} candles, Latest: {open_time}, Price: ${open_price:.4f}, Vol: {volume:.2f}")
                    else:
                        print(f"  ❌ {interval}: Empty data")
                else:
                    print(f"  ❌ {interval}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ {interval}: {str(e)}")
            
            time.sleep(0.1)  # Rate limiting

def test_network_issues():
    """Ağ sorunlarını test et"""
    print("🌐 Network Issues Testi...")
    
    # DNS çözümleme testi
    import socket
    hosts = ['api.binance.com', 'api1.binance.com', 'data-api.binance.vision']
    
    for host in hosts:
        try:
            ip = socket.gethostbyname(host)
            print(f"✅ DNS {host} -> {ip}")
        except Exception as e:
            print(f"❌ DNS {host}: {e}")
    
    # Farklı timeout'larla test
    print("\n⏱️  Timeout Testi:")
    timeouts = [1, 3, 5, 10, 15]
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=10"
    
    for timeout in timeouts:
        try:
            start = time.time()
            response = requests.get(url, timeout=timeout)
            duration = time.time() - start
            print(f"  ✅ Timeout {timeout}s: Success in {duration:.2f}s, Status: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"  ⏰ Timeout {timeout}s: TIMEOUT")
        except Exception as e:
            print(f"  ❌ Timeout {timeout}s: {e}")

def test_rate_limits():
    """Rate limit testi"""
    print("🚦 Rate Limit Testi...")
    
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=5"
    
    for i in range(10):
        try:
            start = time.time()
            response = requests.get(url, timeout=5)
            duration = time.time() - start
            
            # Rate limit headers
            weight = response.headers.get('X-MBX-USED-WEIGHT-1M', 'N/A')
            print(f"  Request {i+1}: Status {response.status_code}, Weight: {weight}, Time: {duration:.2f}s")
            
            if response.status_code == 429:
                print("  ⚠️  Rate limit hit!")
                break
                
        except Exception as e:
            print(f"  ❌ Request {i+1}: {e}")
        
        time.sleep(0.1)

def main():
    print("🚀 BINANCE API DIAGNOSTIC TOOL")
    print("=" * 50)
    
    test_basic_connection()
    test_server_time()
    test_symbol_info()
    test_klines_data()
    test_network_issues()
    test_rate_limits()
    
    print("\n🏁 Test tamamlandı!")

if __name__ == "__main__":
    main()
