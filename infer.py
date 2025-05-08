import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import numpy as np
import matplotlib.pyplot as plt
import config
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import sys
from data import merge_data

def parse_opt():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="checkpoints/100_sport_clf.keras", help="model path(s)")
    parser.add_argument("--source", type=str, default="data/test_images", help="file/dir/URL/glob/screen/0(webcam)")
    opt = parser.parse_args()
    
    return opt

def predict(weight, source):
    # load model
    clf_model = keras.models.load_model(weight)
    # load an image
    img = keras.utils.load_img(path=source, target_size=(config.img_width, config.img_heigth)) # PIL Image
    # convert PIL Image to np array
    img = keras.utils.img_to_array(img)
    input_arr = np.array([img]) # [1, 224, 224]
    predicts = clf_model.predict(input_arr, verbose=0) # [1, 100]
    predicted_class = np.argmax(predicts, axis=1)[0]
    print(f"Label of an image: {config.int2label[predicted_class]}")

if __name__ == "__main__":
    opt = parse_opt()
    predict(**vars(opt))