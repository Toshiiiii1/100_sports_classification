# 100+ sports image classification

This model classifies images into one of 105 sports. It can support more sports in the future. The model was trained using two datasets: one with [100 sports](https://www.kaggle.com/datasets/gpiosenka/sports-classification) and another with [22 sports](https://www.kaggle.com/datasets/rishikeshkonapure/sports-image-dataset). It uses a pre-trained EfficientNetB0 model. The final model reached 98% accuracy and 98% F1 score.

## Dependecies

Python 3.11+ and Keras 3.9+, other dependecies listed in `requirements.txt`.

## Quickstart

#### Installation

```bash
# Clone repo
git clone https://github.com/Toshiiiii1/100_sports_classification.git

# Start virtual eviroment
python -m venv venv
source venv/Scripts/activate

# Check venv activate
which python

# Install required Python libraries
pip install -r requirements.txt

# Download model
python checkpoints/checkpoint_download.py
```

#### Inference
```bash
python infer.py --weight weight-path --source image-path
```

#### Validation

```bash
python val.py --weight weight-path --data data-path --batch 32 --imgz 224
```

#### Training or fine-tuning

```bash
python train.py --model model-path --epoch 10 --batch 64 --lr 1e-5 --n-classes 105 --train-path train-path --test-path test-path --valid-path valid-path
```

Data structure:

```
train set/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg

test set/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg

...
```

#### Merge old data with new data to transfer learning

Upcomming