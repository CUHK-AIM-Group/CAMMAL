import os
import pickle
from tqdm import tqdm
import pandas as pd

root = './adamatch/data/topk_word_list'
# folder_list = ['2023_07_29_21_39_42_keywords_test_top10', '2023_07_29_21_39_42_keywords_train_top10']
# folder_list = ['2023_09_21_19_00_21_newkeywords_test_top10', '2023_09_21_19_00_21_newkeywords_train_top10']
# openI
folder_list = ['2023_10_09_20_02_08_newkeywords_test_top10', '2023_10_09_20_02_08_newkeywords_train_top10']


out_name = folder_list[0].replace('test_', '')
save_root = './adamatch/data/topk_word_list/pickles'
save_path = os.path.join(save_root, out_name + ".pickle")

keywords_dict = {}
# folder_name = folder_list[1]
for folder_name in folder_list:
    print("folder_name:",folder_name)
    folder_path = os.path.join(root, folder_name)
    csv_list = os.listdir(folder_path)
    for csv_name in tqdm(csv_list):
        csv_path = os.path.join(folder_path, csv_name)
        keywords_list = pd.read_csv(csv_path)
        keywords_list = keywords_list['0'].tolist()
        keywords_dict[csv_name.split('.')[0]] = keywords_list


# Store data (serialize)
with open(save_path, 'wb') as handle:
    pickle.dump(keywords_dict, handle)
print("Saved to :", save_path)
print("Finished")