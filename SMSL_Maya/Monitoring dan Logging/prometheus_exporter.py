# prometheus_exporter.py (WORKING VERSION)
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
from flask import Flask, request, Response
import psutil
import time
import threading
import requests
import json

app = Flask(__name__)

# Metrics
PREDICTION_COUNT = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_ERRORS = Counter('model_errors_total', 'Total number of prediction errors')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds',
                                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Memory usage percentage')
MODEL_CONFIDENCE = Gauge('model_confidence_score', 'Average model confidence')
ACTIVE_REQUESTS = Gauge('active_requests', 'Currently active requests')
REQUEST_RATE = Counter('request_rate_total', 'Total requests per second')

# Variables
start_time = time.time()
inference_url = "http://localhost:5000/predict"

def update_system_metrics():
    """Update system metrics periodically"""
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        MODEL_CONFIDENCE.set(0.85)  # Default value
        time.sleep(5)

@app.route('/proxy_predict', methods=['POST'])
def proxy_predict():
    """Proxy endpoint that records metrics and forwards to inference server"""
    ACTIVE_REQUESTS.inc()
    REQUEST_RATE.inc()
    
    start_time = time.time()
    try:
        # Forward request ke inference server
        data = request.get_json()
        response = requests.post(inference_url, json=data, timeout=10)
        latency = time.time() - start_time
        
        # Record metrics
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNT.inc()
        
        # Return response
        return response.json(), response.status_code
        
    except Exception as e:
        PREDICTION_ERRORS.inc()
        return {"error": str(e)}, 500
    finally:
        ACTIVE_REQUESTS.dec()

@app.route('/metrics')
def metrics():
    """Endpoint for Prometheus to scrape"""
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/health')
def health():
    return {"status": "healthy", "metrics_url": "http://localhost:8000/metrics"}

if __name__ == "__main__":
    # Start background thread
    thread = threading.Thread(target=update_system_metrics, daemon=True)
    thread.start()
    
    # Start Prometheus metrics server
    start_http_server(8000)
    
    print("="*60)
    print("PROMETHEUS METRICS EXPORTER (FIXED VERSION)")
    print("="*60)
    print(f"Metrics endpoint: http://localhost:8000/metrics")
    print(f"Proxy endpoint: http://localhost:5001/proxy_predict")
    print(f"Health check: http://localhost:8000/health")
    print(f"\nTo use this exporter, update traffic generator to send requests to:")
    print(f"   http://localhost:5001/proxy_predict (instead of localhost:5000)")
    print("="*60)
    
    # Run Flask proxy
    app.run(port=5001)