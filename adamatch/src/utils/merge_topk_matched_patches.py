
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


root = '~/code/image2text2image/FLIP_medical/data/results/'
# folder_list = ['2023_09_21_19_00_21_patches_test', '2023_09_21_19_00_21_patches_train'] #, '2023_08_14_07_26_14_train']
# openI dataset
folder_list = ['2023_10_09_20_02_08_patches_test', '2023_10_09_20_02_08_patches_train']

out_name = folder_list[0].replace('_test', '')
save_root = '~/code/image2text2image/FLIP_medical/data/topk_patch_list'
#'~code/img2text2img/MGCA/data/topk_patch_list/pickles'
save_path = os.path.join(save_root, out_name + "_matched_top20_patches_for_all_reports.pickle")
save_dict_path = os.path.join(save_root, out_name + "_matched_top20_patches_for_all_reports_dict.pickle")

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

patch_sim_dict = {}
for item in patch_sim_list:
    dicom_id = item['path'].split('/')[-1].split('.')[0]
    vq_code = item['vq_code']
    patch_sim_dict[dicom_id] = vq_code


with open(save_dict_path, 'wb') as handle:
    pickle.dump(patch_sim_dict, handle)
print("Saved to :", save_dict_path)
print("Finished")


'''
filepath = "~code/img2text2img/MGCA/data/topk_patch_list/pickles/2023_08_14_07_26_14_test.pickle"
with open(filepath, "rb") as f:
    data = pickle.load(f)
'''