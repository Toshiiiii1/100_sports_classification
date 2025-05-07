import os
import shutil
import random

def copy_old_dataset(src_folder_path, des_folder_path, subset="train", ratio=1):
    os.makedirs(des_folder_path, exist_ok=True)
    # create subset path
    src_subset_path = os.path.join(src_folder_path, subset) # example: source/train
    des_subset_path = os.path.join(des_folder_path, subset) # example: destination/train
    os.makedirs(des_subset_path, exist_ok=True)
    
    # take each sub folder (label) of source subset path
    for folder_name in os.listdir(src_subset_path):
        src_subfolder_path = os.path.join(src_subset_path, folder_name) # example: source/train/football
        des_subfolder_path = os.path.join(des_subset_path, folder_name) # example: destination/train/football
        os.makedirs(des_subfolder_path, exist_ok=True)
        # get random files or get all files
        if ratio == 1:
            file_names = os.listdir(src_subfolder_path)
        else:
            # file_names = random.choices(os.listdir(src_subfolder_path), k=num_files) # example: [0001.png, 0002.png,...]
            file_names = os.listdir(src_subfolder_path) # example: [0001.png, 0002.png,...]
            file_names = file_names[:int(len(file_names)*ratio)]
        
        for file_name in file_names:
            src_file_path = os.path.join(src_subfolder_path, file_name) # example: source/train/football/001.png
            des_file_path = os.path.join(des_subfolder_path, file_name) # example: destination/train/football/001.png
            # copy file
            shutil.copyfile(src_file_path, des_file_path)

def copy_new_dataset(src_path, des_path, num_files=5):
    # TODO: handle num_files parameter
    
    for folder_name in os.listdir(src_path):
        folder_path = os.path.join(src_path, folder_name)
        # get all image names in sub folder
        img_name = os.listdir(folder_path)
        # shuffle it
        random.shuffle(img_name)
        
        # take 5 images, put them into valid set
        valid_set = img_name[:num_files]
        # take 5 images, put them into test set
        test_set = img_name[num_files:num_files*2]
        # the remain is train set
        train_set = img_name[num_files*2:]
        
        # copy each subset to its corresponding subset in destination folder
        for subset, img_set in zip(("train", "valid", "test"), (train_set, valid_set, test_set)):
            des_folder_path = os.path.join(os.path.join(des_path, subset), folder_name)
            os.makedirs(des_folder_path, exist_ok=True)
            
            for img_name in img_set:
                img_src_path = os.path.join(folder_path, img_name)
                des_img_path = os.path.join(des_folder_path, img_name)
                shutil.copyfile(img_src_path, des_img_path)

if __name__ == "__main__":
    sport_22_path = "data/raw/22_sports_from_Konapure"
    sport_100_path = "data/raw/100_sports"
    des_path = "data/preprocessed/dataset_merged_1"
    
    # copy train set from 100 sports to preprocessed folder
    copy_old_dataset(src_folder_path=sport_100_path, des_folder_path=des_path, ratio=1)
    
    # copy entire valid set and test set from 100 sports to preprocessed folder
    copy_old_dataset(src_folder_path=sport_100_path, des_folder_path=des_path, subset="valid", ratio=1)
    copy_old_dataset(src_folder_path=sport_100_path, des_folder_path=des_path, subset="test", ratio=1)
    
    # split new dataset to train set, valid set and test set. Copy all subset to megred folder
    copy_new_dataset(src_path=sport_22_path, des_path=des_path)