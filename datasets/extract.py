from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os

from ast import literal_eval

image_and_annotations_df = pd.read_csv("mydata.csv")


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

  ctr = 0
  for row in df_one_img:
    img = crop_image(path, row)
    cars_image.append(img)
    outputfilename = os.path.join(path, "extracted_files", f"myoutputfile_{ctr}.jpg")
    ctr += 1

    #img.save(outputfilename)

  return cars_image

test_file = os.path.join(cars_folder, "train", "DOH_10_mp4-5_jpg.rf.1b99964f5eada026f922d242423833c8.jpg")
print(test_file)

x = extract_cars("DOH_10_mp4-5_jpg.rf.1b99964f5eada026f922d242423833c8.jpg", image_and_annotations_df)
print(x)
