import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import csv
import pandas as pd

df = pd.read_csv("../data/data.csv")



print(df['planet'])

y = df['planet']
x = df["pixels"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

num_planets = len(np.unique(y))
num_samples = len(y)

num_hidden = 10

model = tf.keras.Sequential()

#read after this
model.add(layers.Dense(num_hidden, activation="relu"))

model.add(layers.Dense(num_planets))

#model

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
	metrics=['accuracy'])

epochs = 500

batch_size = num_samples//5
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

print(history)