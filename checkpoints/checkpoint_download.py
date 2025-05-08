import gdown
import argparse

def parse_opt():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="", help="GG Drive url")
    parser.add_argument("--save_path", type=str, default="checkpoints/model.keras", help="save model")
    opt = parser.parse_args()
    
    return opt

def download_model(url, save_path):
    # download model from Google Drive
    gdown.download(url=url, output=save_path, quiet=False, fuzzy=True)
    print("Model downloaded successfully.")

if __name__ == "__main__":
    opt = parse_opt()
    download_model(**vars(opt))