
import pickle
# import matplotlib.pyplot as plt
import numpy as np
import math
# import cv2 
import os
from tqdm import tqdm

# out_dir = '~code/img2text2img/MGCA/data/topk_patch_list'

# file_root = "~code/img2text2img/MGCA/data/results/2023_08_14_07_26_14_test"
# #"~code/img2text2img/MGCA/data/results/2023_07_29_21_39_42_test"
# file_list = os.listdir(file_root)

# for pickle_filename in tqdm(file_list):
#     if not 'pickle' in pickle_filename:
#         continue
#     filepath = os.path.join(file_root, pickle_filename)
#     print(filepath)
#     # filepath="~code/img2text2img/MGCA/data/results/2023_07_29_21_39_42/cuda:0_idx_0.pickle"
#     with open(filepath, "rb") as f:
#         data = pickle.load(f)


root = './adamatch/data/results/'
# folder_list = ['2023_09_21_19_00_21_test', '2023_09_21_19_00_21_train'] #, '2023_08_14_07_26_14_train']
# folder_list = ['2023_08_14_07_26_14_test', '2023_08_14_07_26_14_train'] #, '2023_08_14_07_26_14_train']

# for openI dataset
# folder_list includes the folders for matched keypatches
folder_list = ['2023_10_09_20_02_08_test', '2023_10_09_20_02_08_train']


out_name = folder_list[0].replace('_test', '')
save_root = './adamatch/data/topk_patch_list/pickles'
save_path = os.path.join(save_root, out_name + ".pickle")

patch_sim_list = []
# folder_name = folder_list[1]
for folder_name in folder_list:
    print("folder_name:",folder_name)
    folder_path = os.path.join(root, folder_name)
    file_list = os.listdir(folder_path)
    for pickle_filename in tqdm(file_list):
        if not 'pickle' in pickle_filename:
            continue
        filepath = os.path.join(folder_path, pickle_filename)
        # print(filepath)
        # filepath="~code/img2text2img/MGCA/data/results/2023_07_29_21_39_42/cuda:0_idx_0.pickle"
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            patch_sim_list.extend(data)


# Store data (serialize)
with open(save_path, 'wb') as handle:
    pickle.dump(patch_sim_list, handle)
print("Saved to :", save_path)
print("Finished")
