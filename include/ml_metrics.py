import requests
import json
from datetime import datetime

class GrafanaMetrics:
    def __init__(self, prometheus_url, username, api_key):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.username = username
        self.api_key = api_key
    
    def push_training_metrics(self, run_id, epoch, metrics):
        """Push logs to Grafana Cloud Loki"""
        import base64
        
        # Use the provided Loki URL directly
        loki_url = self.prometheus_url
        
        timestamp_ns = str(int(datetime.now().timestamp() * 1000000000))
        
        # Format for Loki (logs)
        streams = []
        for metric_name, value in metrics.items():
            log_line = f"TRAINING_METRIC: {metric_name}={value} epoch={epoch} run_id={run_id}"
            streams.append({
                "stream": {
                    "job": "airflow",
                    "metric": metric_name,
                    "model": "hurricane_damage"
                },
                "values": [[timestamp_ns, log_line]]
            })
        
        payload = {"streams": streams}
        
        # Basic auth
        auth_str = f"{self.username}:{self.api_key}"
        auth_b64 = base64.b64encode(auth_str.encode()).decode()
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {auth_b64}'
        }
        
        try:
            print(f"TRAINING_METRIC: {json.dumps(metrics)} epoch={epoch}")
            
            response = requests.post(
                loki_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                print(f"✅ Logs sent to Grafana successfully")
            else:
                print(f"❌ Grafana logs failed: {response.status_code} - {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ Grafana logs failed: {e}")
        
        return True