import os
import shutil
import numpy as np
import time
from PIL import Image
import json

def extract_label(file_path):
    filename = os.path.basename(file_path)
    filename_no_ext = filename.replace('.json', '')
    parts = filename_no_ext.split('_')

    # Find the index of the UUID (has 4 hyphens)
    uuid_index = next((i for i, part in enumerate(parts) if part.count('-') == 4), len(parts))

    # Special case: if the first part is "Biopsie", join it with the next part using a space
    if parts[0] == "Biopsie":
        label = f"{parts[0]} {parts[1]}"
    else:
        # General case: join all parts from index 1 to the UUID as the label
        label = ' '.join(parts[1:uuid_index])

    return label

path = "/scratch/nmoreau/glom_classification/"


time_start = time.time()

path_images = os.path.join(path, "/glom_extracted_glom_images/")
path_split = "splits_kidney.json"
train_save_path = os.path.join(path, 'pipnet_dataset/train/')
val_save_path = os.path.join(path, 'pipnet_dataset/val/')
test_save_path = os.path.join(path, 'pipnet_dataset/test/')

with open(path_split, 'r') as f:
    json_split = json.load(f)
train_list = json_split["split_1"] + json_split["split_2"]
val_list = json_split["split_3"]
test_list = json_split["test"]

os.makedirs(train_save_path)
os.makedirs(train_save_path + "/class1/")
os.makedirs(train_save_path + "/class2/")
os.makedirs(val_save_path)
os.makedirs(val_save_path + "/class1/")
os.makedirs(val_save_path + "/class2/")
os.makedirs(test_save_path)
os.makedirs(test_save_path + "/class1/")
os.makedirs(test_save_path + "/class2/")

for file in os.listdir(path_images):
    name = path_images + "/" + file
    glom_class = -1
    split = -1
    wsi_name = extract_label(file)
    if file.find("_type_1") != -1:
        glom_class = "class1"
    elif file.find("_type_2") != -1:
        glom_class = "class2"
    elif file.find("_type_3") != -1:
        glom_class = "class2"

    if wsi_name in train_list:
        split = "train"
    elif wsi_name in val_list:
        split = "val"
    elif wsi_name in test_list:
        split = "test"

    if glom_class != -1 and split != -1:
        new_name = path + "/pipnet_dataset/" + split + "/" + glom_class + "/" + file
        shutil.copyfile(name, new_name)

time_end = time.time()
print('CUB200, %s!' % (time_end - time_start))
