
import pickle
# import matplotlib.pyplot as plt
import numpy as np
import math
# import cv2 
import os
from tqdm import tqdm
import torch
import _pickle as cPickle
def findElements(lst1, lst2):
    return list(map(lst1.__getitem__, lst2))

sim_topk = 1000
word_token_topk = 20 #5
threshold = 0.6
vq_path = "~dataset/x_ray/mimic_xray/info/mimiccxr_vqgan1024_res256_3e_codebook_indices.pickle"
filepath = "~code/img2text2img/MGCA/data/topk_patch_list/pickles/2023_08_14_07_26_14.pickle"
save_path = os.path.join("~code/img2text2img/MGCA/data/topk_patch_list", "top"+str(sim_topk)+"_patches_list.pickle")
with open(filepath, "rb") as f:
    data = pickle.load(f)

sim_list = []
word_token_num_list = []
path_list = []
for i in tqdm(range(len(data))):
    sim_list.append(torch.tensor(data[i]['sim']).unsqueeze(0)) # patch_token_num x word_token_num
    word_token_num_list.append(torch.tensor(1 - data[i]['special_token_mask']).sum()) # 
    path_list.append(data[i]['path'].split('/')[-1].split('.')[0])

word_token_num_list = torch.tensor(word_token_num_list)
# print("word_token_num_list:",word_token_num_list.shape)
# concat 
sim_list = torch.cat(sim_list, dim=0) # bs x patch_token_num x word_token_num
sim_val = sim_list.max(1).values # bs x word_token_num
sim_val = sim_val.sum(1) # bs x 1 
# batch["cap_lens"] : bs,
sim_val = sim_val / word_token_num_list
# print("sim_list:",sim_list.shape)

_, indices = torch.sort(sim_val, descending=True)

# topk_sim_val = sorted_sim_val[:sim_topk]
# print(topk_sim_val.mean(), topk_sim_val.max(), topk_sim_val.min())

topk_indices = indices[:sim_topk]
# print(type(topk_indices),topk_indices.shape)
sorted_sim_list = sim_list[topk_indices]
# sorted_path_list = path_list[topk_indices]
sorted_path_list = findElements(path_list, topk_indices)
# max_sim_list, max_sim_indices = torch.max(sorted_sim_list, dim=1) #  bs x patch_token_num x word_token_num ->  bs x 1 x word_token_num

# _, max_word_sim_indices = torch.max(max_sim_list, dim=2) # bs x 1 x word_token_num
# topk_word_sim_indices = max_word_sim_indices[:word_token_topk]

with open(vq_path, "rb") as f:
    cxr_vq = cPickle.load(f)

all_matched_vq_dict = {}
for i in tqdm(range(sorted_sim_list.shape[0])):
    each_sample = sorted_sim_list[i] # 1 x patch_token_num x word_token_num
    # print("each_sample:",each_sample.shape)
    cxr_id = sorted_path_list[i]
    # print("cxr_id:",cxr_id)
    # print("has cxr_id:", cxr_id in cxr_vq)
    vq_code = cxr_vq[cxr_id] # (256,)
    # print("vq_code:",vq_code)
    # max
    max_patch_sim_list, max_patch_sim_indices = torch.max(each_sample, dim=1) # 1 x 1 x word_token_num
    max_patch_sim_list = max_patch_sim_list.squeeze() # word_token_num
    max_patch_sim_indices = max_patch_sim_indices.squeeze() # word_token_num
    # print("max_patch_sim_list:",max_patch_sim_list)
    # print("max_patch_sim_indices:", max_patch_sim_indices)
    # max_vq_code_list = vq_code[max_patch_sim_indices]
    max_vq_code_list = findElements(vq_code, max_patch_sim_indices)
    # print("max_vq_code_list:",max_vq_code_list)
    # sorted
    sorted_max_patch_sim_list, sorted_max_patch_sim_indices = torch.sort(max_patch_sim_list, descending=True) # word_token_num
    # sorted_max_vq_code_list = max_vq_code_list[sorted_max_patch_sim_indices]
    sorted_max_vq_code_list = findElements(max_vq_code_list, sorted_max_patch_sim_indices)

    # vq_code[]
    # print("sorted_max_patch_sim_list:",sorted_max_patch_sim_list)
    # print("sorted_max_patch_sim_indices:",sorted_max_patch_sim_indices)
    # print("sorted_max_vq_code_list:",sorted_max_vq_code_list)
    final_vqcode_list = []
    for code in sorted_max_vq_code_list:
        if not code in final_vqcode_list:
            final_vqcode_list.append(code)
    # print("final_vqcode_list:",final_vqcode_list)
    # print("set - vqcode:", set(sorted_max_vq_code_list))
    # print("more than threshold:",threshold)
    # indices_th = sorted_max_patch_sim_list>threshold
    # sorted_max_patch_sim_list_th = sorted_max_patch_sim_list[indices_th]
    # sorted_max_vq_code_list_th = findElements(sorted_max_vq_code_list, indices_th)
    # print("indices_th:",indices_th)
    # print("sorted_max_patch_sim_list_th:",sorted_max_patch_sim_list_th)
    # print("sorted_max_vq_code_list_th:",sorted_max_vq_code_list_th)
    # exit(0)
    all_matched_vq_dict[cxr_id] = final_vqcode_list


# Store data (serialize)
with open(save_path, 'wb') as handle:
    pickle.dump(all_matched_vq_dict, handle)
print("Saved to :", save_path)
print("Finished")




'''
filepath = "top1000_patches_list.pickle"
with open(filepath, "rb") as f:
    data = pickle.load(f)
'''