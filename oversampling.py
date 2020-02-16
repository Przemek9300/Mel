import pandas as pd
from random import sample
import os
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import shutil

CLASSES_LABEL = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def load_csv(path='HAM10000_metadata.csv'):
    data = pd.read_csv(path)
    dicts = data.to_dict('records')
    return dicts


def preapereDir(dir='ready'):
    os.makedirs(dir)
    for item in CLASSES_LABEL:
         os.makedirs(f'./{dir}/{item}')
def splitImages(data):
    for type in data:
        if not os.path.exists(f'data/{type["dx"]}'):
            os.makedirs(f'data/{type["dx"]}')
        shutil.copy(f'./HAM10000_images/{type["image_id"]}.jpg',
                    f'./data/{type["dx"]}', follow_symlinks=False)
def preprocessing(dir, num_samples):
    for item in CLASSES_LABEL:
        save_path = f'./{dir}'

        batch_size = 50
        src_files = os.listdir(save_path)

        data_gen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=(0.9,1.1),
            fill_mode='nearest')

        aug_datagen = data_gen.flow_from_directory(f'./data/',
                                                   save_to_dir=save_path,
                                                   save_format='jpg',
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   batch_size=batch_size)
        num_files = len(os.listdir(save_path))

        num_batches = int(np.ceil((num_samples - num_files) ))
        print(f'{num_files} {num_batches}')
        if num_batches > 0:
            for i in range(0, 10):
                print(i)
                imgs, labels = next(aug_datagen)
                print(imgs , labels)

data = load_csv()
# splitImages(data)
# preapereDir()
# preprocessing('ready',3000)
