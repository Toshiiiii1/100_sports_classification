import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import config
from keras.src.utils import image_dataset_from_directory
import warnings
warnings.filterwarnings("ignore")
    
def parse_opt():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--weight", type=str, default="checkpoints/105_sport_clf.keras", help="model path(s)")
    parser.add_argument("--data", type=str, default="data/test_images", help="test folder")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--imgz", type=int, default=config.IMG_SIZE[0], help="image size")
    
    opt = parser.parse_args()
    
    return opt

def val(opt):
    test_ds = image_dataset_from_directory(
        opt.data,
        image_size=(opt.imgz, opt.imgz),
        batch_size=opt.batch,
        label_mode="categorical",
    )
    
    # load model
    clf_model = keras.models.load_model(opt.weight)
    
    class_names = test_ds.class_names
    
    # create lists to store true labels and predictions
    y_true = []
    y_pred = []

    # iterate through the test dataset
    for images, labels in test_ds:
        # get predictions
        predictions = clf_model.predict(images, verbose=0)
        
        # convert predictions to class indices
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(predicted_classes)

    # calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    plt.savefig("cm.png")

    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    opt = parse_opt()
    val(opt)