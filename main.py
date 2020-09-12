from keras.models import load_model
import pandas as pd
import os
import shutil
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


img_width, img_height = 224, 224
train_samples = 3200
batch_size = 10
train_data_dir = './ready'


# imports the mobilenet model and discards the last 1000 neuron layer.
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# we add dense layers so that the model can w moQre complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
# final layer with softmax activation
preds = Dense(7, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=preds)


for layer in model.layers[:-23]:
    layer.trainable = False
# model = load_model('m.hdf5', compile=False)
model.compile(optimizer='Adam',  loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.3)  # set validation split

train_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    'data',  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # set as validation data

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=1)
model.save('m.hdf5')
model_json = model.to_json()


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
