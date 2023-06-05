import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

import matplotlib.pyplot as plt

import tensorflow as tf

data_dir = "../data/planets"
batch_size = 50

data = tf.keras.utils.image_dataset_from_directory(data_dir)

#define batch iterator
data_iterator = data.as_numpy_iterator()

#scale image pixels
data = data.map(lambda x, y : (x/255, y))



batch = data_iterator.next()

"""
#matplotlib code that allows to view and troubleshoot images
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
	ax[idx].imshow(img.astype(int))
	ax[idx].title.set_text(batch[1][idx])

plt.show()

"""

train_size = int(len(data)*0.5)
validation_size = int(len(data)*0.4)+1
test_size = int(len(data)*0.1)+1

train = data.take(train_size)
validation = data.skip(train_size).take(validation_size)
test = data.skip(train_size + validation_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (9,9), 1, activation="relu", input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (9,9), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(16, (9,9), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="relu"))

model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.summary()


history = model.fit(train, epochs=20, validation_data=validation)

fig = plt.figure()
plt.plot(history.history['loss'], color="teal", label="loss")
plt.plot(history.history["val_loss"], color="orange", label="val_loss")
fig.suptitle("Loss")
plt.legend(loc="upper left")
plt.show()