from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
from os import path
import shutil
import pandas as pd
from PIL import Image
import glob

if __name__ == "__main__":
    label_table = pd.read_csv('./CSFID/label_table.csv')
    period_label = pd.read_csv('./CSFID/period_label.csv')
    tracks_name_number = pd.read_csv('./CSFID/tracks_name_number.csv')
    # CSV to dict
    label_table = dict(label_table.values)
    period_label = dict(period_label.values)
    tracks_name_number = dict(tracks_name_number.values)

    CATEGORIES = ['not_periodic','periodic']

    # print(len(label_table))
    # print(len(period_label))
    # print(len(tracks_name_number))


    cwd = os.getcwd()
    print(cwd)
    src = "./CSFID/tracks_cropped"

    dst = cwd + "/data/"
    periodic_dst = cwd + "/data/periodic/"
    not_periodic_dst = cwd + "/data/not_periodic/"
    print(dst)
    train_dst = dst + 'train/'
    test_dst = dst + 'test/'

    train_periodic_dst = train_dst + "periodic/"
    train_not_periodic_dst = train_dst + "not_periodic/"

    test_periodic_dst = test_dst + "periodic/"
    test_not_periodic_dst = test_dst + "not_periodic/"
    if not os.path.exists(dst):
        os.makedirs(dst)

        # Create sub-folders.
        for sub in CATEGORIES:
            os.mkdir(os.path.join(cwd, "data", sub))


    read_tracks_cropped()
    # read_original_tracks()
    read_from_references()