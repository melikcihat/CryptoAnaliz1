import requests
import json

print("Testing Binance API...")

try:
    # Test 1: Ping
    r = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
    print(f"Ping: {r.status_code}")
    
    # Test 2: Server time
    r = requests.get('https://api.binance.com/api/v3/time', timeout=10)
    print(f"Time: {r.status_code}")
    
    # Test 3: BTC data
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 5}
    r = requests.get(url, params=params, timeout=10)
    print(f"BTC Data: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"Data length: {len(data)}")
        if data:
            print(f"First candle: {data[0]}")
    else:
        print(f"Error: {r.text}")

except Exception as e:
    print(f"Exception: {e}")
