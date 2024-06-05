from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from ast import literal_eval


image_and_annotations_df = pd.read_csv("image_and_annotations.csv")
image_df = pd.read_csv("image.csv")

cars_folder = "Car Counting.v1i.coco"

def crop_image(image_path, bbox):
  img = Image.open(image_path)

  left, top, width, height = literal_eval(bbox)
  cropped = img.crop((left, top, left + width, top + height))

  return cropped

def extract_cars(image_name, dataframe):
  cars_image = []

  df_one_img = dataframe[dataframe['file_name'] == image_name]["bbox"]

  car_path = os.path.join(cars_folder, "train")
  path = os.path.join(car_path, image_name)

  for row in df_one_img:
    img = crop_image(path, row)
    cars_image.append(img)

  return cars_image

test_file = os.path.join(cars_folder, "train", "DOH_10_mp4-5_jpg.rf.1b99964f5eada026f922d242423833c8.jpg")
print(test_file)

print("Starting")

cropped_images = [extract_cars(img, image_and_annotations_df) for img in image_df["file_name"]]

start = time.time()

ctr = 0
for arr in cropped_images:
  if ctr > 2500: break

  for img in arr:
    img = img.resize((128, 128))
    img.save(f"extracted_scaled/{ctr}.jpg")
    ctr += 1

  
end = time.time()
print(end - start)
print("done")