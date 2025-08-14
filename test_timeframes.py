#!/usr/bin/env python3
"""
45 dakika ve 3 saatlik timeframe test scripti
"""

import pandas as pd
import yfinance as yf
import requests

def test_yahoo_45m():
    """Yahoo Finance 45 dakika test"""
    print("🔍 Testing Yahoo Finance 45m...")
    try:
        t = yf.Ticker("BTC-USD")
        # 45m için 15m çekip resample
        df = t.history(period="3d", interval="15m", progress=False)
        if not df.empty:
            print(f"✅ Got 15m data: {len(df)} bars")
            
            # 45m resample
            df_45m = pd.DataFrame()
            df_45m["Open"] = df["Open"].resample("45min").first()
            df_45m["High"] = df["High"].resample("45min").max()
            df_45m["Low"] = df["Low"].resample("45min").min()
            df_45m["Close"] = df["Close"].resample("45min").last()
            df_45m["Volume"] = df["Volume"].resample("45min").sum()
            df_45m = df_45m.dropna()
            
            print(f"✅ 45m resampled: {len(df_45m)} bars")
            print(f"Latest 45m price: ${df_45m['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No 15m data from Yahoo")
            return False
    except Exception as e:
        print(f"❌ Yahoo 45m error: {e}")
        return False

def test_yahoo_3h():
    """Yahoo Finance 3 saat test"""
    print("\n🔍 Testing Yahoo Finance 3h...")
    try:
        t = yf.Ticker("BTC-USD")
        # 3h için 1h çekip resample  
        df = t.history(period="7d", interval="1h", progress=False)
        if not df.empty:
            print(f"✅ Got 1h data: {len(df)} bars")
            
            # 3h resample
            df_3h = pd.DataFrame()
            df_3h["Open"] = df["Open"].resample("3h").first()
            df_3h["High"] = df["High"].resample("3h").max()
            df_3h["Low"] = df["Low"].resample("3h").min()
            df_3h["Close"] = df["Close"].resample("3h").last()
            df_3h["Volume"] = df["Volume"].resample("3h").sum()
            df_3h = df_3h.dropna()
            
            print(f"✅ 3h resampled: {len(df_3h)} bars")
            print(f"Latest 3h price: ${df_3h['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No 1h data from Yahoo")
            return False
    except Exception as e:
        print(f"❌ Yahoo 3h error: {e}")
        return False

def test_binance_3h():
    """Binance 3 saat test"""
    print("\n🔍 Testing Binance 3h...")
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '3h',
            'limit': 20
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"✅ Binance 3h: {len(data)} bars")
                last_candle = data[-1]
                close_price = float(last_candle[4])
                print(f"Latest Binance 3h price: ${close_price:.2f}")
                return True
            else:
                print("❌ Empty data from Binance")
                return False
        else:
            print(f"❌ Binance API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Binance 3h error: {e}")
        return False

def test_binance_45m():
    """Binance 45 dakika test (15m resample)"""
    print("\n🔍 Testing Binance 45m (via 15m resample)...")
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '15m',
            'limit': 100  # 45m için daha fazla 15m data
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"✅ Got 15m data: {len(data)} bars")
                
                # Convert to DataFrame
                rows = []
                for k in data:
                    timestamp = pd.to_datetime(int(k[0]), unit='ms')
                    rows.append([
                        timestamp, float(k[1]), float(k[2]), 
                        float(k[3]), float(k[4]), float(k[5])
                    ])
                
                df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df.set_index('Date', inplace=True)
                
                # 45m resample
                df_45m = pd.DataFrame()
                df_45m["Open"] = df["Open"].resample("45min").first()
                df_45m["High"] = df["High"].resample("45min").max()
                df_45m["Low"] = df["Low"].resample("45min").min()
                df_45m["Close"] = df["Close"].resample("45min").last()
                df_45m["Volume"] = df["Volume"].resample("45min").sum()
                df_45m = df_45m.dropna()
                
                print(f"✅ Binance 45m resampled: {len(df_45m)} bars")
                if not df_45m.empty:
                    print(f"Latest Binance 45m price: ${df_45m['Close'].iloc[-1]:.2f}")
                    return True
                else:
                    print("❌ Empty after resample")
                    return False
            else:
                print("❌ Empty data from Binance")
                return False
        else:
            print(f"❌ Binance API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Binance 45m error: {e}")
        return False

def main():
    print("🚀 TIMEFRAME TEST SCRIPT")
    print("=" * 40)
    
    results = {
        "Yahoo 45m": test_yahoo_45m(),
        "Yahoo 3h": test_yahoo_3h(),
        "Binance 3h": test_binance_3h(),
        "Binance 45m": test_binance_45m()
    }
    
    print("\n📊 RESULTS:")
    print("=" * 40)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    working_count = sum(results.values())
    print(f"\nWorking sources: {working_count}/{len(results)}")
    
    if working_count > 0:
        print("🎉 At least one source is working!")
    else:
        print("🚨 All sources failed!")

if __name__ == "__main__":
    main()
