Grafana Cloud integration for hurricane-damage project
===============================================

This folder contains a minimal Grafana dashboard JSON and instructions to visualize training metrics (accuracy and loss per epoch) pushed by the training job to a Prometheus Pushgateway.

Overview (recommended path)
- The training task (Airflow) uses a Keras callback to push per-epoch metrics to a Prometheus Pushgateway.
- Run a Prometheus server or Grafana Agent that scrapes the Pushgateway and then forwards metrics to Grafana Cloud (or scrape Pushgateway directly if reachable).
- Import the provided dashboard JSON into Grafana Cloud to visualize metrics.

Quick setup
1. Grafana Cloud
   - Create an account at https://grafana.com/cloud.
   - Create a Prometheus-compatible remote_write endpoint or get your Grafana Cloud metrics/Prometheus details.

2. Pushgateway
   - If you don't have infrastructure to host a Pushgateway, you can run one locally with Docker:

     ```powershell
     docker run -d -p 9091:9091 prom/pushgateway
     ```

   - Set the `PUSHGATEWAY_URL` environment variable in your Airflow environment (or in the DAG via Airflow Connections/Variables) to point to the Pushgateway, e.g. `http://pushgateway:9091`.

3. Grafana Agent / Prometheus
   - Configure Prometheus or Grafana Agent to scrape the Pushgateway job metrics and forward them to Grafana Cloud.
   - Example Grafana Agent scrape config (example snippet) — set remote_write to Grafana Cloud credentials:

     ```yaml
     metrics:
       global:
         scrape_interval: 15s
       configs:
         - name: pushgateway-scrape
           scrape_interval: 15s
           static_configs:
             - targets: ['pushgateway:9091']
           metrics_path: /metrics
           relabel_configs: []
     ```

4. Import dashboard
   - In Grafana Cloud, go to Dashboards → Import and upload `dashboard.json`.

Notes & troubleshooting
- If Airflow tasks are ephemeral and cannot reach your Pushgateway, consider using a hosted remote_write client in your environment or push directly to Grafana Cloud using a Prometheus remote_write client.
- The training callback will not fail the training if pushing fails; it prints a warning instead.

Metrics exported
- train_accuracy_percent (Gauge, labelled by epoch)
- val_accuracy_percent
- train_loss
- val_loss

If you'd like, I can also:
- Add a small Docker Compose file to run a local Pushgateway + Grafana for testing
- Create a more advanced dashboard with confusion matrix heatmap (requires pushing per-class counts)
