import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import numpy as np
from keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils import class_weight
import argparse
import config
from keras.src.utils import image_dataset_from_directory
import warnings
warnings.filterwarnings("ignore")

def train(opt):
    # prepare dataset
    train_ds = image_dataset_from_directory(
        opt.train_path,
        image_size=(opt.imgz, opt.imgz),
        batch_size=opt.batch,
        label_mode="categorical",
    )
        
    val_ds = image_dataset_from_directory(
        opt.valid_path,
        image_size=(opt.imgz, opt.imgz),
        batch_size=opt.batch,
        label_mode="categorical",
    )
    
    # apply augmentation for train set
    if opt.augmentation:
        img_augmentation_layers = [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ]
        
        def data_augmentation(images):
            for layer in img_augmentation_layers:
                images = layer(images)
            return images
        
        train_ds.map(
            lambda img, label: (data_augmentation(img), label)
        )
        
    # get all labels in train set
    all_label = []
    for _, label in train_ds.unbatch(): # label: one-hot style label such as  [0 0 0 0 0 0 0 1]
        class_index = np.argmax(label)
        all_label.append(class_index)
    all_label = np.array(all_label)
    
    # get unique label
    classes = np.unique(all_label)
    
    # get class weights for unbalanced data (if any)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=all_label
    )
    
    # mapping classes to its corespond weight
    class_weight_dict = dict(zip(classes, class_weights))
    
    # TODO: add learning rate schedule and early stopping, maybe save model per epoch or save best model
    
    # choose optimizer
    if opt.optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=opt.lr)
    elif opt.optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=opt.lr)
    elif opt.optimizer == "AdamW":
        optimizer = keras.optimizers.AdamW(learning_rate=opt.lr)
    elif opt.optimizer == "RMSProp":
        optimizer = keras.optimizers.RMSprop(learning_rate=opt.lr)
    
    # whether using pre-train model or not
    if opt.model:
        print("Use pre-trained")
        # load pre-train model
        model = keras.saving.load_model(opt.model)
        # compare model output and number of classes of new dataset
        n_output = model.output.shape[-1]
        if opt.n_classes != n_output:
            print("Modify clf layer")
            # take the backbone of pre-trained model
            backbone = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
            # freeze all layers of the backbone
            backbone.trainable = False
            # add new clf layer
            x = backbone.output
            new_output = layers.Dense(opt.n_classes, activation="softmax", name="prediction")(x)
            model = keras.models.Model(inputs=backbone.input, outputs=new_output, name="ModEfficientNet")
            
        # unfreeze final block in backbone and clf layer
        for layer in model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    else:
        print("Use model from scratch")
        # define model
        model = EfficientNetB0(
            include_top=True,
            weights=None,
            classes=opt.n_classes,
            input_shape=(opt.imgz, opt.imgz, 3),
        )
        
    # compile model
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
        
    # train model
    hist = model.fit(train_ds, epochs=opt.epoch, validation_data=val_ds, class_weight=class_weight_dict)
    
    # TODO: create loss, accuracy during training chart
    
def parse_opt():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default=None, help="weights path")
    parser.add_argument("--imgz", type=int, default=config.IMG_SIZE[0], help="image size")
    parser.add_argument("--augmentation", action="store_true", help="whether apply augmentation or not")
    parser.add_argument("--epoch", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    parser.add_argument("--n-classes", type=int, default=100, help="number of classes")
    parser.add_argument("--train-path", type=str, default="", help="train set path")
    parser.add_argument("--test-path", type=str, default="", help="test set path")
    parser.add_argument("--valid-path", type=str, default="", help="valid set path")
    parser.add_argument("--save", type=str, default="checkpoints/model.keras", help="save path")
    
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    train(opt)