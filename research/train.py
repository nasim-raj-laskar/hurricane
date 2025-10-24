import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Sequential,Input                                                                                       # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, Rescaling        # type: ignore
from tensorflow.keras.callbacks import EarlyStopping                                                                                # type: ignore
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Paths
dirs = {
    "train": r'X:\nasim_xhqpjmy\Code\MLops\hurricane-damage\dataset\train_another',
    "val": r'X:\nasim_xhqpjmy\Code\MLops\hurricane-damage\dataset\validation_another',
    "test": r'X:\nasim_xhqpjmy\Code\MLops\hurricane-damage\dataset\test'
}

IMG_SIZE = (128, 128)
BATCH = 32

# Dataset loader
def load_ds(path):
    return tf.keras.preprocessing.image_dataset_from_directory(path, image_size=IMG_SIZE, batch_size=BATCH)

train_ds, val_ds, test_ds = map(load_ds, [dirs["train"], dirs["val"], dirs["test"]])

# Model
model = Sequential([
    Input(shape=(96, 96, 3)),
    Rescaling(1./255, input_shape=(96, 96, 3)),
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train + Evaluate
history = model.fit(train_ds, validation_data=val_ds, epochs=5, 
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc*100:.2f}%")