"""Prometheus helper for training metrics.

Provides a Keras callback that pushes per-epoch metrics to a Prometheus Pushgateway.

Usage:
  from utils.metrics import TrainingMetricsCallback

  cb = TrainingMetricsCallback(pushgateway_url=os.getenv('PUSHGATEWAY_URL'))
  model.fit(..., callbacks=[cb])

If PUSHGATEWAY_URL is not provided, the callback will only update an internal registry
and will not attempt to push (useful for local testing).
"""
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import tensorflow as tf
from typing import Optional


class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, pushgateway_url: Optional[str] = None, job: str = "hurricane_training", run_id: Optional[str] = None):
        super().__init__()
        self.pushgateway_url = pushgateway_url
        self.job = job
        self.run_id = run_id or "unknown"
        self.registry = CollectorRegistry()

        # Diagnostic print so Airflow task logs show whether callback was constructed
        print(f"[METRICS] TrainingMetricsCallback initialized. pushgateway_url={self.pushgateway_url}, job={self.job}")

        # Gauges labelled by epoch and run_id so you can query/plot each epoch/run as a timeseries
        self.g_train_acc = Gauge('train_accuracy_percent', 'Training accuracy (percent)', ['epoch', 'run_id'], registry=self.registry)
        self.g_val_acc = Gauge('val_accuracy_percent', 'Validation accuracy (percent)', ['epoch', 'run_id'], registry=self.registry)
        self.g_train_loss = Gauge('train_loss', 'Training loss', ['epoch', 'run_id'], registry=self.registry)
        self.g_val_loss = Gauge('val_loss', 'Validation loss', ['epoch', 'run_id'], registry=self.registry)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        e = str(epoch + 1)

        # Logs may contain accuracy as 0-1; convert accuracy to percent for easier display
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if acc is not None:
            self.g_train_acc.labels(epoch=e, run_id=self.run_id).set(float(acc) * 100)
        if val_acc is not None:
            self.g_val_acc.labels(epoch=e, run_id=self.run_id).set(float(val_acc) * 100)
        if loss is not None:
            self.g_train_loss.labels(epoch=e, run_id=self.run_id).set(float(loss))
        if val_loss is not None:
            self.g_val_loss.labels(epoch=e, run_id=self.run_id).set(float(val_loss))

        if self.pushgateway_url:
            try:
                print(f"[METRICS] Pushing metrics to {self.pushgateway_url} for job={self.job}, run_id={self.run_id}, epoch={e}")
                push_to_gateway(self.pushgateway_url, job=self.job, registry=self.registry)
                print(f"[METRICS] Push successful for job={self.job}, run_id={self.run_id}, epoch={e}")
            except Exception:
                # Push should never crash training - swallow and continue
                import traceback
                print(f"[WARN] failed to push metrics to {self.pushgateway_url}")
                traceback.print_exc()
