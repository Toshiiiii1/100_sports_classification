import gdown
import config

if __name__ == "__main__":
    gdown.download(url=config.model_ggdrive_path, output=config.model_path, quiet=False, fuzzy=True)
    print("Model downloaded successfully.")