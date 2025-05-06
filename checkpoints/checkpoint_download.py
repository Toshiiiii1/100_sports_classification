import gdown
from config import model_url

if __name__ == "__main__":
    # download model from Google Drive
    gdown.download(url=model_url, output="checkpoints/100_sport_clf.keras", quiet=False, fuzzy=True)
    print("Model downloaded successfully.")