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

def read_tracks_cropped():

    for filename in glob.glob('CSFID/tracks_cropped/*.jpg'):  # assuming jpg
        file_key = int(filename.split('\\')[-1].split('.')[0])
        file_class = period_label[file_key]

        # print(filename)
        # print('key', file_key, 'class:', period_label[file_key])
        if file_class == 0:
            shutil.copy(filename, test_not_periodic_dst)
        else:
            shutil.copy(filename, test_periodic_dst)


def read_original_tracks():
    tmp = tracks_name_number
    # print(tmp)

    for filename in glob.glob('CSFID/tracks_original/*.jpg'):  # assuming jpg
        file_key = str(filename.split('\\')[-1])
        # file_class = tracks_name_number[file_key]
        try:
            print(file_key)
            print(tmp[file_key])
            img_key = int(tmp[file_key].split('.')[0])
            print('key', img_key, 'class:', period_label[img_key])
            file_class = period_label[img_key]
            if file_class == 0:
                shutil.copy(filename, test_not_periodic_dst)
            else:
                shutil.copy(filename, test_periodic_dst)

            del(tmp[file_key])

        except:
            print("erreer",filename)
            print(tmp)



def read_from_references():
    tmp = label_table
    for filename in glob.glob('CSFID/references/*.png'):  # assuming jpg
        file_key = str(filename.split('\\')[-1])
        img_name = int(file_key.split('.')[0])
        if img_name in tmp.values():
            img_key = get_key(tmp,img_name)
            print(img_name,img_key)
            now = datetime.now().strftime("%f")
            try:
                print('key', img_key, 'class:', period_label[img_key])
                file_class = period_label[img_key]
                if file_class == 0:
                    shutil.copy(filename, train_not_periodic_dst+str(now)+".png")
                else:
                    shutil.copy(filename, train_periodic_dst+str(now)+".png")


            except:
                print("erreer", filename)
                print(tmp)

# function to return key for any value
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"

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