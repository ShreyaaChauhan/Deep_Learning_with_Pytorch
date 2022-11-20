# responsible for dividing and structuring
#  the dataset into training and validation
from __future__ import annotations

import os
import shutil

import config as config
import numpy as np
from imutils import paths

os.makedirs(config.TRAIN, exist_ok=True)
os.makedirs(config.VAL, exist_ok=True)


def copy_images(image_Paths: list, folder: str):
    shutil.rmtree(folder)
    '''
        This function copies image from one
        folder to another folder
    '''
    os.makedirs(folder, exist_ok=True)
    for path in image_Paths:
        imageName = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[-2]
        labelFolder = os.path.join(folder, label)

        os.makedirs(labelFolder, exist_ok=True)

        destination = os.path.join(labelFolder, imageName)
        shutil.copy(path, destination)


# Load and shuffle the image path
print('[INFO] loading image paths...')
imagePaths = list(paths.list_images(config.TOBACCO_DATASET_PATH))
# print(len(imagePaths))
np.random.shuffle(imagePaths)

# split Image ptahs into train and val set
valImagesLen = int(len(imagePaths)*config.VAL_SPLIT)
trainImagesLen = len(imagePaths) - valImagesLen
trainPaths = imagePaths[:trainImagesLen]
valPaths = imagePaths[trainImagesLen:]
print('[INFO] Copying Image from sorce folder to destination')
copy_images(trainPaths, config.TRAIN)
copy_images(valPaths, config.VAL)
print(f'[INFO] {len(list(paths.list_images(config.TRAIN)))} \
images in TRAIN folder \n \
[INFO] {len(list(paths.list_images(config.VAL)))} \
images in Test folder')
# print(len(list(set(trainPaths+valPaths))))
