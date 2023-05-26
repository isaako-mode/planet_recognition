import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import csv
import pandas as pd
import ast

import tensorflow as tf

data_dir = "../data/planets"
image_size = (224, 224)
batch_size = 32

# Create an ImageDataGenerator instance
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data from the directory
dataset = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

