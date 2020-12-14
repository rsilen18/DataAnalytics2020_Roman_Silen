# hurr_class.py
# hurricane image classification using TensorFlow

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# create the dataset
import pathlib
data_dir = pathlib.Path("./images")
image_count = len(list(data_dir.glob('*/*.jp*g')))
print(image_count)


# create data loader
# parameters for data loader
batch_size = 16
img_height = 1024
img_width = 1024
color_mode = "rgb"
img_depth = 3
if (color_mode == "grayscale"):
    img_depth = 1
# split into training/validation sets (80/20 split)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode=color_mode)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode=color_mode)
# print class names - should be 'falsehurr' and 'hurricane'
class_names = train_ds.class_names
print(class_names)


# display one image with colorbar
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
      for i in range(1):
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.colorbar()
        plt.grid = False


# configure dataset for performance
# buffered prefetching to yield data from disk w/o I/O blocking
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# data augmentation
# take existing data and use random transformations to expose model
# to more aspects of data and avoid overfitting
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, img_depth)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
# visualize a few augmented examples
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
      for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


# create/compile neural network model
# different cases for with and w/o Augmentation
# vary Dropout proportions
num_classes = 2
dropout = 0.2
aug = False

if (aug == False):
	model = Sequential([
	  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, img_depth)),
	  layers.Conv2D(16, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Conv2D(32, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Conv2D(64, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Dropout(dropout),
	  layers.Flatten(),
	  layers.Dense(128, activation='relu'),
	  layers.Dense(num_classes)
	])
else:
	model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, img_depth, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, img_depth, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, img_depth, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(dropout),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


# train the model
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# create plots of in-sample error and validation error
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(2,1,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
if (aug):
    title = 'Training and Validation Accuracy (' + str(img_height) + 'x' + str(img_width) + ' - ' + color_mode + ', dropout = ' + str(dropout) + ' w/ augmentation)'
else:
    title = 'Training and Validation Accuracy (' + str(img_height) + 'x' + str(img_width) + ' - ' + color_mode + ', dropout = ' + str(dropout) + ')'
plt.title(title)

plt.subplot(2,1,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
if (aug):
    title = 'Training and Validation Loss (' + str(img_height) + 'x' + str(img_width) + ' - ' + color_mode + ', dropout = ' + str(dropout) + ' w/ augmentation)'
else:
    title = 'Training and Validation Loss (' + str(img_height) + 'x' + str(img_width) + ' - ' + color_mode + ', dropout = ' + str(dropout) + ')'
plt.title(title)
plt.show()
