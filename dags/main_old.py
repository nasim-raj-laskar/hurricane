from airflow import DAG                                                        # type: ignore
from airflow.operators.python import PythonOperator                            # type: ignore
from airflow.hooks.base import BaseHook                                        # type: ignore
from airflow.models import Variable                                            # type: ignore
from datetime import datetime, timedelta                            
import os, zipfile, boto3                                                            

# CONFIG
S3_BUCKET = 'hurricane-damage-data'
DATA_DIR = '/tmp/data'
LOCAL_ZIP = os.path.join(DATA_DIR, 'dataset.zip')
IMG_SIZE = (128, 128)
BATCH = 4

default_args = {
    'owner': 'hurricane-team',
    'start_date': datetime(2025, 10, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

#----------------------------------------------------------task-1----------------------------------------------------------
def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = BaseHook.get_connection('aws_default')

    s3 = boto3.client(
        's3',
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=conn.extra_dejson.get('region_name', 'eu-north-1')
    )

    s3.download_file(S3_BUCKET, 'dataset.zip', LOCAL_ZIP)
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    return os.path.join(DATA_DIR, 'dataset')


#----------------------------------------------------------task-2----------------------------------------------------------
def load_datasets(**context):
    root = context["task_instance"].xcom_pull(task_ids='download_data')

    dirs = {
        "train": os.path.join(root, "train_another"),
        "val": os.path.join(root, "validation_another"),
        "test": os.path.join(root, "test")
    }

    return dirs


#----------------------------------------------------------task-3----------------------------------------------------------
def build_and_train(**context):
    import mlflow
    import tensorflow as tf
    from tensorflow.keras import Sequential, Input #type:ignore
    from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D,SeparableConv2D, GlobalAveragePooling2D, Rescaling) #type:ignore
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
    import warnings
    import os
    import gc
    warnings.filterwarnings("ignore")

    conn = BaseHook.get_connection("mlflow_dagshub")

    os.environ["MLFLOW_TRACKING_USERNAME"] = conn.login
    os.environ["MLFLOW_TRACKING_PASSWORD"] = conn.password
    os.environ["MLFLOW_TRACKING_URI"] = conn.extra_dejson.get(
        "tracking_uri", "https://dagshub.com/nasim-raj-laskar/hurricane.mlflow"
    )

    print("[INFO] Using Dagshub MLflow Auth:")
    print(" - User:", os.getenv("MLFLOW_TRACKING_USERNAME"))
    print(" - URI:", os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("hurricane_damage_training_v5")

    #sanity check
    print("MLFLOW_TRACKING_USERNAME:", os.getenv("MLFLOW_TRACKING_USERNAME"))
    print("MLFLOW_TRACKING_PASSWORD:", os.getenv("MLFLOW_TRACKING_PASSWORD")[:4], "***")
    print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))

    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["train"], image_size=IMG_SIZE, batch_size=BATCH
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["val"], image_size=IMG_SIZE, batch_size=BATCH
    )

    #Model
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Rescaling(1./255),
        Conv2D(8, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(16, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        SeparableConv2D(32, 3, activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #MLflow 
    with mlflow.start_run(run_name="hurricane_training") as run:
        # Push metrics to Prometheus Pushgateway. Configure PUSHGATEWAY_URL in Airflow
        try:
            from utils.metrics import TrainingMetricsCallback
        except Exception as e:
            print(f"[METRICS] Could not import TrainingMetricsCallback: {e}")
            import traceback
            traceback.print_exc()
            TrainingMetricsCallback = None

        callback_list = []
        if TrainingMetricsCallback is not None:
            # Prefer environment variable, fall back to an Airflow Variable named PUSHGATEWAY_URL
            push_url_env = os.getenv('PUSHGATEWAY_URL')
            push_url_var = None
            try:
                push_url_var = Variable.get('PUSHGATEWAY_URL', default_var=None)
            except Exception:
                push_url_var = None

            push_url = push_url_env or push_url_var
            source = 'env' if push_url_env else ('airflow.Variable' if push_url_var else 'none')
            print(f"[METRICS] PUSHGATEWAY_URL resolved from {source}: {push_url}")

            cb = TrainingMetricsCallback(pushgateway_url=push_url, job=f"hurricane_{run.info.run_id}", run_id=run.info.run_id)
            callback_list.append(cb)
        else:
            print("[METRICS] TrainingMetricsCallback not available; no metrics will be pushed to Pushgateway")

        print(f"[METRICS] callback_list size before training: {len(callback_list)}")

        history = model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=callback_list)

        # Log to MLflow
        mlflow.log_params({
            "epochs": 2,
            "batch_size": BATCH,
            "img_size": IMG_SIZE,
            "optimizer": "Adam",
            "learning_rate": 0.001
        })
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1] * 100)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1] * 100)
        mlflow.log_metric("train_loss", history.history['loss'][-1] * 100)
        mlflow.log_metric("val_loss", history.history['val_loss'][-1] * 100)

        
        # Local backup logs
        print("=== ML TRAINING METRICS ===")
        for epoch in range(len(history.history['accuracy'])):
            metrics = {
                'epoch': epoch + 1,
                'train_accuracy': round(history.history['accuracy'][epoch] * 100, 2),
                'val_accuracy': round(history.history['val_accuracy'][epoch] * 100, 2),
                'train_loss': round(history.history['loss'][epoch], 4),
                'val_loss': round(history.history['val_loss'][epoch], 4)
            }
            print(f"TRAINING_METRIC: {metrics}")
        print("=== END METRICS ===")

        model.save('/tmp/trained_model.h5')
        mlflow.log_artifact('/tmp/trained_model.h5', artifact_path="model")

        # Push run_id for next tasks
        context['task_instance'].xcom_push(key='run_id', value=run.info.run_id)
    tf.keras.backend.clear_session()
    gc.collect()
    return True


#----------------------------------------------------------task-4----------------------------------------------------------
def evaluate_model(**context):
    import tensorflow as tf
    import mlflow
    import numpy as np
    from airflow.hooks.base import BaseHook #type:ignore 
    from utils.metrics import push_additional_metrics, start_continuous_mock_gpu_metrics, push_classification_metrics
    import os

    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')
    run_id = context['task_instance'].xcom_pull(task_ids='build_and_train', key='run_id')

    mlflow_conn = BaseHook.get_connection('mlflow_dagshub')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_conn.login
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_conn.password
    mlflow.set_tracking_uri(mlflow_conn.extra_dejson['tracking_uri'])

    model = tf.keras.models.load_model('/tmp/trained_model.h5')

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    loss, acc = model.evaluate(test_ds)
    
    # Get predictions for classification metrics
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("Test Accuracy", acc)
        mlflow.log_metric("Test Loss", loss)
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

    #  Push to Prometheus
    try:
        from airflow.models import Variable #type:ignore
        push_url_env = os.getenv('PUSHGATEWAY_URL')
        push_url_var = None
        try:
            push_url_var = Variable.get('PUSHGATEWAY_URL', default_var=None)
        except Exception:
            push_url_var = None
        push_url = push_url_env or push_url_var
        print(f"[METRICS] Using Pushgateway URL: {push_url}")

        params = {
            "batch_size": BATCH,
            "learning_rate": 0.001,
            "epochs": 2,
            "img_size": IMG_SIZE,
            "optimizer": "Adam"
        }

        if push_url:
            push_additional_metrics(push_url, run_id, acc, loss, params)
            push_classification_metrics(push_url, run_id, y_true, y_pred, ['no_damage', 'damage'])
            start_continuous_mock_gpu_metrics(push_url, run_id)
        else:
            print("[METRICS] Skipped pushing additional metrics (no URL found).")

    except Exception as e:
        print("[WARN] Failed to push test metrics to Prometheus:", e)

    return {'accuracy': acc, 'loss': loss}



#----------------------------------------------------------task-5----------------------------------------------------------
def save_model_s3(**context):
    import boto3, mlflow, os
    from datetime import datetime
    from airflow.hooks.base import BaseHook #type:ignore

    run_id = context['task_instance'].xcom_pull(task_ids='build_and_train', key='run_id')

    mlflow_conn = BaseHook.get_connection('mlflow_dagshub')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_conn.login
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_conn.password
    mlflow.set_tracking_uri(mlflow_conn.extra_dejson['tracking_uri'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_model = '/tmp/trained_model.h5'
    renamed_path = f"/tmp/hurricane_{timestamp}.h5"

    if os.path.exists(tmp_model):
        os.rename(tmp_model, renamed_path)
    else:
        raise FileNotFoundError(f"{tmp_model} not found before upload!")

    s3 = boto3.client('s3')
    s3.upload_file(renamed_path, S3_BUCKET, f"models/hurricane_{timestamp}.h5")

    s3_uri = f"s3://{S3_BUCKET}/models/hurricane_{timestamp}.h5"

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("s3_model_path", s3_uri)
        mlflow.log_artifact(renamed_path)

    # Cleanup
    os.remove(renamed_path)
    return s3_uri


# #----------------------------------------------------------task-6----------------------------------------------------------
# def cleanup_resources(**context):
#     import shutil
    
#     print("[CLEANUP] Starting resource cleanup...")
    
#     # Remove dataset directory
#     if os.path.exists(DATA_DIR):
#         try:
#             shutil.rmtree(DATA_DIR)
#             print(f"[CLEANUP] Removed {DATA_DIR}")
#         except Exception as e:
#             print(f"[CLEANUP] Failed to remove {DATA_DIR}: {e}")
    
#     # Remove any remaining model files in /tmp
#     tmp_models = [f for f in os.listdir('/tmp') if f.startswith('hurricane_') and f.endswith('.h5')]
#     for model_file in tmp_models:
#         try:
#             os.remove(os.path.join('/tmp', model_file))
#             print(f"[CLEANUP] Removed /tmp/{model_file}")
#         except Exception as e:
#             print(f"[CLEANUP] Failed to remove {model_file}: {e}")
    
#     print("[CLEANUP] Cleanup completed")



# ------------------------DAG DEFINITION----------------------------
dag = DAG(
    'hurricane_damage_training_mono',
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=['ml', 'tensorflow', 'dagshub', 'mlflow']
)

t1 = PythonOperator(task_id='download_data', python_callable=download_data, dag=dag)
t2 = PythonOperator(task_id='load_datasets', python_callable=load_datasets, dag=dag)
t3 = PythonOperator(task_id='build_and_train', python_callable=build_and_train, dag=dag)
t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)
t5 = PythonOperator(task_id='save_model_s3', python_callable=save_model_s3, dag=dag)
# t6 = PythonOperator(task_id='cleanup_resources', python_callable=cleanup_resources, dag=dag, trigger_rule='all_done')

t1 >> t2 >> t3 >> t4 >> t5 #>>t6
