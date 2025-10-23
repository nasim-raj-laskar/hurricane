import tensorflow as tf
import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing import image_dataset_from_directory                                                          #ignore
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Rescaling,RandomFlip,RandomZoom,RandomRotation     #ignore
from tensorflow.keras.models import Sequential
import keras
print("TensorFlow version:", tf.__version__)