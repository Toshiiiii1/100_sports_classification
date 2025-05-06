import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import config
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    clf_model = keras.models.load_model(config.model_path)
    img = keras.utils.load_img(path="data/test_images/basketball_test_1.jpg", target_size=(config.img_width, config.img_heigth))
    img = keras.utils.img_to_array(img)
    input_arr = np.array([img])
    predicts = clf_model.predict(input_arr, verbose=0)
    predicted_class = np.argmax(predicts, axis=1)
    print(config.int2label[predicted_class[0]])