import pandas as pd
from random import sample
import os
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

CLASSES_LABEL = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def dirPrepare():
    os.mkdir('train')
    os.mkdir('val')
    for label in CLASSES_LABEL:
        os.mkdir(f'train/{label}')
        os.mkdir(f'val/{label}')


def init():
    df = pd.read_csv('clean.csv')   
    y = df['dx']
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=101, stratify=y)

    # Print the shape of the training and validation split
    print(df_train.shape)
    print(df_val.shape)

    # Find the number of values in the training and validation set
    df_train['dx'].value_counts()
    df_val['dx'].value_counts()
    df.set_index('image_id', inplace=True)
    folder_1 = os.listdir('ham10000_images')
    train_list = list(df_train['image_id'])
    val_list = list(df_val['image_id'])


    print(f'Train data: {len(train_list)}')
    for index,image in enumerate(train_list):

        fname = image + '.jpg'
        label = df.loc[image, 'dx']

        if fname in folder_1:
            # source path to image
            src = os.path.join('HAM10000_images', fname)
            # destination path to image
            dst = os.path.join('train', label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
            print("%.0f%%" % (100*index/len(train_list)))

    # Transfer the validation images
    print(f'Train data: {len(val_list)}')

    for image in val_list:

        fname = image + '.jpg'
        label = df.loc[image, 'dx']

        if fname in folder_1:
            # source path to image
            src = os.path.join('HAM10000_images', fname)
            # destination path to image
            dst = os.path.join('val', label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
            print("%.0f%%" % (100*index/len(val_list)))
def dataAugamation():
    class_list = ['mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    for item in class_list:

        # Create a temporary directory for the augmented images


        # Create a directory within the base dir to store images of the same class
        img_dir = os.path.join(aug_dir, item)
        os.mkdir(img_dir)

        # Choose a class
        img_class = item

        # List all the images in the directory
        img_list = os.listdir('val/' + img_class)

        # Copy images from the class train dir to the img_dir
        for fname in img_list:
            # source path to image
            src = os.path.join('val/' + img_class, fname)
            # destination path to image
            dst = os.path.join(img_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

        # point to a dir containing the images and not to the images themselves
        path = aug_dir
        save_path = 'val/' + img_class

        # Create a data generator to augment the images in real time
        datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            # brightness_range=(0.9,1.1),
            fill_mode='nearest')

        batch_size = 50

        aug_datagen = datagen.flow_from_directory(path,
                                                save_to_dir=save_path,
                                                save_format='jpg',
                                                target_size=(224, 224),
                                                batch_size=batch_size)

        # Generate the augmented images and add them to the training folders
        num_aug_images_wanted = 500  # total number of images we want to have in each class
        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

        # run the generator and create about 6000 augmented images
        for i in range(0, num_batches):
            imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
def cleanCsv():
    raw_data = pd.read_csv('HAM10000_metadata.csv')
    clean_data = raw_data.drop_duplicates(subset='lesion_id')
    clean_data.to_csv('clean.csv')
dataAugamation()