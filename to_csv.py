from os import listdir
from scipy.io import loadmat
import pandas as pd

crowd_folder = "datasets/ShanghaiTech_Crowd_Counting_Dataset/Shanghai"
cars_folder = "datasets/Car Counting.v1i.coco/"

path = "datasets/ShanghaiTech_Crowd_Counting_Dataset/Shanghai/test_data/ground_truth"

crowd_info_train = [mat for mat in listdir(crowd_folder + "/train_data/ground_truth")]
crowd_info_test = [mat for mat in listdir(crowd_folder + "/test_data/ground_truth")]

print("Crowd Info Training:", len(crowd_info_train))
print("Crowd Info Testing:", len(crowd_info_test))

print(crowd_info_train[0].split('.')[0])

for matFile in crowd_info_train:
    fileName = matFile.split('.')[0]
    # print(crowd_folder + "/train_data/ground_truth/" + crowd_info_train[0])
    mat_frame_test = loadmat(crowd_folder + "/train_data/ground_truth/" + fileName)
    # print(mat_frame_test.keys())
    x = mat_frame_test["image_info"][0][0][0][0][0]
    # col1 = mat_frame_test["image_info"]
    # print(col1)
    df = pd.DataFrame(x)
    df.to_csv(f"mat_to_csv/crowd_info_train/{fileName}.csv")

for matFile in crowd_info_test:
    fileName = matFile.split('.')[0]
    # print(crowd_folder + "/train_data/ground_truth/" + crowd_info_train[0])
    mat_frame_test = loadmat(crowd_folder + "/train_data/ground_truth/" + fileName)
    # print(mat_frame_test.keys())
    x = mat_frame_test["image_info"][0][0][0][0][0]
    # col1 = mat_frame_test["image_info"]
    # print(col1)
    df = pd.DataFrame(x)
    df.to_csv(f"mat_to_csv/crowd_info_test/{fileName}.csv")


# print(x[0])

# mat = {k:v for k, v in mat_frame_test.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()}) # compatible for both python 2.x and python 3.x

# data.to_csv("test.csv")