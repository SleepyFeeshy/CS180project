from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
from ast import literal_eval


def crop_images_from_csv(img_anno_csv, img_csv, src_path, dst_path, img_size):
  image_and_annotations_df = pd.read_csv(img_anno_csv)
  image_df = pd.read_csv(img_csv)

  def crop_image(image_path, bbox):
    img = Image.open(image_path)

    left, top, width, height = literal_eval(bbox)
    cropped = img.crop((left, top, left + width, top + height))

    return cropped

  def extract_cars(image_name, dataframe):
    cars_image = []

    df_one_img = dataframe[dataframe['file_name'] == image_name]["bbox"]

    car_path = src_path
    path = os.path.join(car_path, image_name)

    for row in df_one_img:
      img = crop_image(path, row)
      cars_image.append(img)

    return cars_image

  cropped_images = [extract_cars(img, image_and_annotations_df) for img in image_df["file_name"]]

  ctr = 0

  print("Start")

  for arr in cropped_images:
    for img in arr:
      height, width = img.height, img.width
      max_size = max(height, width)
      r = max_size / img_size
      new_width = int(width / r)
      new_height = int(height / r)
      new_size = (new_width, new_height)
  
      img = img.resize(new_size, Image.Resampling.LANCZOS)

      new_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
      new_img[0:new_height, 0:new_width] = img
      new_img = Image.fromarray(new_img)
      new_img.save(f"{dst_path}/{ctr}.jpg")

      ctr += 1

  print("Done")

crop_images_from_csv("image_and_annotations.csv", "image.csv", "Car Counting.v1i.coco/train", "train_128x128", 128)

crop_images_from_csv("test_img_and_anno.csv", "test_img.csv", "Car Counting.v1i.coco/test", "test_128x128", 128)

