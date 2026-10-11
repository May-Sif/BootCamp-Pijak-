# generate_traffic.py (UPDATED - kirim ke proxy)
import requests
import time
import random

# Ganti URL ke proxy exporter (bukan langsung ke inference)
PROXY_URL = "http://localhost:5001/proxy_predict"

sample_features = [
    [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3],
    [53, 1, 0, 140, 203, 1, 0, 155, 1, 3.1, 0, 0, 3],
    [70, 1, 0, 145, 174, 0, 1, 125, 1, 2.6, 0, 0, 3],
]

def send_prediction(features):
    data = {"features": features}
    start = time.time()
    
    try:
        response = requests.post(PROXY_URL, json=data, timeout=5)
        latency = time.time() - start
        return response.status_code, latency, response.json()
    except Exception as e:
        return 500, 0, {"error": str(e)}

print("="*50)
print("TRAFFIC GENERATOR (via Metrics Proxy)")
print(f"Sending to: {PROXY_URL}")
print("="*50)

count = 0
try:
    while True:
        features = random.choice(sample_features)
        interval = random.uniform(1, 3)
        
        status, latency, result = send_prediction(features)
        count += 1
        
        pred = result.get('prediction', '?') if status == 200 else 'ERROR'
        print(f"[{count}] Status: {status} | Latency: {latency:.3f}s | Prediction: {pred} | Next: {interval:.1f}s")
        
        time.sleep(interval)
        
except KeyboardInterrupt:
    print(f"\nTotal requests: {count}")