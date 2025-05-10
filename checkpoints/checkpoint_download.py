import gdown
import config
import sys

if __name__ == "__main__":
    gdown.download(url=config.model_ggdrive_path, output="checkpoints/clf_checkpoint.keras", quiet=False, fuzzy=True)
    print("Model downloaded successfully.")