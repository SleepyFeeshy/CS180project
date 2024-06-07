from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

from ast import literal_eval
import numpy as np

def sort_class(image_annotation_csv, src_path, dst_path, start_filename=0):
    image_and_annotations_df = pd.read_csv(image_annotation_csv)

    for i in range(len(os.listdir(src_path))):
        category = image_and_annotations_df[image_and_annotations_df['id'] == start_filename + i]["category_id"].item()
        print(category)
        src_img = f"{start_filename + i}.jpg"
        shutil.move(os.path.join(src_path, src_img), f"{dst_path}/{category}")

    print("Done")

# sort_class("image_and_annotations.csv", "train_2100_128x128/class", "train_2100_128x128")

# sort_class("test_img_and_anno.csv", "test_300_128x128/class", "test_300_128x128")

sort_class("image_and_annotations.csv", "valid_600_128x128/class", "valid_600_128x128", 2100)