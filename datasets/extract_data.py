from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from ast import literal_eval
import numpy as np

def extract_image(image_annotation_csv, src_path, dst_path, start, end):
    image_and_annotations_df = pd.read_csv(image_annotation_csv)

    print("Starting")

    startTime = time.time()

    for id in image_and_annotations_df["id"]:
        if id < start: continue
        if id >= end: break

        image_name = image_and_annotations_df[image_and_annotations_df['id'] == id]["file_name"].item()

        car_path = os.path.join(src_path)
        print(image_name)
        path = os.path.join(car_path, image_name)

        img = Image.open(path)
        img.save(f"{dst_path}/{image_name}.jpg")

    endTime = time.time()
    print(endTime - startTime)
    print("done")

#extract_image("img_ann_noid.csv", "Car Counting.v1i.coco/train", "big_train_2100", 0, 2100) # training dataset

#extract_image("img_ann_noid.csv", "Car Counting.v1i.coco/train", "big_valid_600", 2100, 2700) # validation dataset

#extract_image("img_ann_noid.csv", "Car Counting.v1i.coco/train", "big_test_300", 2700, 3000) # testing dataset
