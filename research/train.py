# Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'                                                                                            # Suppress TensorFlow logging
import tensorflow as tf             
from tensorflow.keras import Sequential,Input                                                                                       # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, Rescaling        # type: ignore
from tensorflow.keras.callbacks import EarlyStopping                                                                                # type: ignore
import warnings  
import boto3
import zipfile
from datetime import datetime               
from dotenv import load_dotenv                                                                                                   # Load environment variables                                                                                                    # Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)                                                                             

load_dotenv()

# AWS S3 Configuration
S3_BUCKET = 'hurricane-damage-data'
S3_KEY = 'dataset.zip'
DATA_DIR = '/tmp/data'  
LOCAL_ZIP = os.path.join(DATA_DIR, 'dataset.zip')

# Ensure /tmp/data exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download dataset from S3
def download_and_extract_s3_dataset():
    print("Downloading dataset from S3...")
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET, S3_KEY, LOCAL_ZIP)
    print("Download complete!")

    print("Extracting dataset...")
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete!")

    # Return extracted base directory
    extracted_root = os.path.join(DATA_DIR, 'dataset')
    return {
        "train": os.path.join(extracted_root, "train_another"),
        "val": os.path.join(extracted_root, "validation_another"),
        "test": os.path.join(extracted_root, "test")
    }


# Parameters
IMG_SIZE = (128, 128)
BATCH = 32

# Dataset loader
def load_ds(path):
    return tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=IMG_SIZE, batch_size=BATCH
    )

# #Augmentation 
# def augment(image, label):
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_brightness(image, max_delta=0.1)
#     # optional zoom simulation (light)
#     scale = tf.random.uniform([], 0.9, 1.1)
#     new_size = tf.cast(tf.convert_to_tensor(IMG_SIZE, dtype=tf.float32) * scale, tf.int32)
#     image = tf.image.resize(image, new_size)
#     image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])
#     return image, label

# # Apply augmentation only to training set
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = (train_ds
#             .map(augment, num_parallel_calls=AUTOTUNE)
#             .cache()
#             .shuffle(1000)
#             .prefetch(buffer_size=AUTOTUNE))

# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Define Model
def build_cnn(input_shape=(128, 128, 3)):
    model = Sequential([
        Input(shape=input_shape), 
        Rescaling(1./255, input_shape=input_shape),
        # First conv block
        Conv2D(8, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        #Second conv block 
        Conv2D(16, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        # Third conv block — depthwise-separable
        tf.keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        #dense head
        Dense(32, activation='relu'),
        Dropout(0.2),
        #Output
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

#Define training 
def train_model(train_ds, val_ds, epochs=5):
    model = build_cnn()
    print(model.summary())

    es = EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[es],
        verbose=1
    )
    return model, history

#Start training
def train():
    model, history = train_model(train_ds, val_ds, epochs=5)
    return model,history


# Evaluate
def evaluate(model, test_ds):
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    return loss, acc

#Save model
def save_model(model, path="hurricane.h5"):
    model.save(path)
    print(f"Model saved as {path}")

#Save model to S3 with timestamp
def save_model_to_s3(model, bucket=S3_BUCKET):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"hurricane_{timestamp}.h5"
    model.save(model_path)
    
    s3 = boto3.client('s3')
    s3.upload_file(model_path, bucket, f"models/{model_path}")
    print(f"Model uploaded to s3://{bucket}/models/{model_path}")
    
    os.remove(model_path)  # Clean up local file

# Main
if __name__ == "__main__":
    dirs = download_and_extract_s3_dataset()
    train_ds, val_ds, test_ds = map(load_ds, [dirs["train"], dirs["val"], dirs["test"]])

    model, history = train_model(train_ds, val_ds, epochs=5)
    evaluate(model, test_ds)
    save_model(model, path="hurricane.h5")
    save_model_to_s3(model)