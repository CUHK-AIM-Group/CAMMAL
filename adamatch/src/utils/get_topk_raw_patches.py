
import pickle
# import matplotlib.pyplot as plt
import numpy as np
import math
import cv2 
import os
from tqdm import tqdm
import torch
import _pickle as cPickle
def findElements(lst1, lst2):
    return list(map(lst1.__getitem__, lst2))

'''def convert_idx(indice, num_p_m, num_p_vq, p_m_size, p_vq_size):
    # compute indice coordinate in large patch map
    i = indice // num_p_m
    j = indice % num_p_m
    ## left upp corner
    y_left = i * p_m_size
    x_left = j * p_m_size
    ## right bottom corner
    y_right = (i+1) * p_m_size
    x_right = (j+1) * p_m_size

    # get indice coordinate in small patch map
    ## position of left upp corner 
    i0 = y_left // p_vq_size
    i0_rest = y_left % p_vq_size
    j0 = x_left // p_vq_size
    j0_rest = x_left % p_vq_size
    ## position of right bottom corner 
    i1 = y_right // p_vq_size
    i1_rest = y_right % p_vq_size
    j1 = x_right // p_vq_size
    j1_rest = x_right % p_vq_size

    quarter = p_vq_size/4
    half = p_vq_size/2
    three = quarter + half 

    out_indice = []
    if i0_rest <= quarter or j0_rest <= quarter:
        # (i0, j0)
        out_indice.append(i0*num_p_vq + j0)
    elif (i0_rest <= three and i0_rest > quarter) or (j0_rest <= three and j0_rest > quarter):
        # (i0, j0)
        # (i1, j1)
        # (i0, j1)
        # (i1, j0)
        out_indice.append(i0*num_p_vq + j0)
        out_indice.append(i1*num_p_vq + j1)
        out_indice.append(i0*num_p_vq + j1)
        out_indice.append(i1*num_p_vq + j0)
    elif i0_rest > three or j0_rest > three:
        out_indice.append(i1*num_p_vq + j1)
    #  
    return out_indice'''

from PIL import Image

def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img

def get_imgs(img_path, scale, crop_size):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
    img = Image.fromarray(x).convert("RGB")
    # centercrop
    width, height = img.size   # Get dimensions

    new_width = new_height = crop_size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    # if transform is not None:
    #     img = transform(img)

    return img

crop_size = 224
img_size = 256
patch_size = 16
num_p_m = 14
num_p_vq= 16
p_m_size = 256 / 14
p_vq_size = 256 /16

sim_topk = 1000
word_token_topk = 20 #5
threshold = 0.6
vq_path = "~dataset/x_ray/mimic_xray/info/mimiccxr_vqgan1024_res256_3e_codebook_indices.pickle"
filepath = "~code/img2text2img/MGCA/data/topk_patch_list/pickles/2023_08_14_07_26_14.pickle"
save_path = os.path.join("~code/img2text2img/MGCA/data/topk_patch_list", "top"+str(sim_topk)+"_patches_list.pickle")
with open(filepath, "rb") as f:
    data = pickle.load(f)

indice_mapping_path = '~code/img2text2img/MGCA/data/topk_patch_list/indice_map.pickle'
with open(indice_mapping_path, "rb") as f:
    indice_mapping_dict = pickle.load(f)

sim_list = []
word_token_num_list = []
path_list = []
for i in tqdm(range(len(data))):
    sim_list.append(torch.tensor(data[i]['sim']).unsqueeze(0)) # patch_token_num x word_token_num
    word_token_num_list.append(torch.tensor(1 - data[i]['special_token_mask']).sum()) # 
    path_list.append(data[i]['path'])#.split('/')[-1].split('.')[0])
    # path_list.append(data[i]['path'].split('/')[-1].split('.')[0])

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
    img_path = sorted_path_list[i]
    
    # img = cv2.imread(img_path, 0) # h, w, ch
    img = get_imgs(img_path, img_size, crop_size)
    # print("img:",img.size,type(img))
    img = np.array(img) # h, w, ch
    # print("img:",img.shape)
    img = np.transpose(img, (2,0,1)) # ch, h, w
    # print("img-2:",img.shape)
    img = torch.tensor(img)
    # img = np.reshape(img, ) # ch, 
    img_patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size) # ch x n_h x n_w x p_h x p_w 
    # print("img_patches:",img_patches.shape) # ch x 14 x 14 x 16 x 16
    ch, n_h, n_w, p_h, p_w = img_patches.shape
    img_patches = img_patches.reshape([ch, n_h*n_w, p_h, p_w])

    cxr_id = img_path.split('/')[-1].split('.')[0]
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
    # max_vq_code_list = findElements(vq_code, max_patch_sim_indices)

    max_patch_sim_indices = max_patch_sim_indices.numpy().tolist()
    max_vq_code_list =[]
    for j in range(len(max_patch_sim_indices)):
        idx = max_patch_sim_indices[j]
        tmp_indice_list = indice_mapping_dict[idx]
        tmp_vq_code_list = findElements(vq_code, tmp_indice_list)
        max_vq_code_list.append(tmp_vq_code_list)
    # get raw patches
    max_raw_patches = img_patches[:, max_patch_sim_indices, :, :]

    # print("max_vq_code_list:",max_vq_code_list)
    # sorted
    sorted_max_patch_sim_list, sorted_max_patch_sim_indices = torch.sort(max_patch_sim_list, descending=True) # word_token_num
    # sorted_max_vq_code_list = max_vq_code_list[sorted_max_patch_sim_indices]

    topk_sorted_max_patch_sim_indices = sorted_max_patch_sim_indices[:word_token_topk]

    sorted_max_vq_code_list = findElements(max_vq_code_list, topk_sorted_max_patch_sim_indices)

    # get sorted raw patches
    sorted_max_raw_patches = max_raw_patches[:, topk_sorted_max_patch_sim_indices, :, :]

    # vq_code[]
    # print("sorted_max_patch_sim_list:",sorted_max_patch_sim_list)
    # print("sorted_max_patch_sim_indices:",sorted_max_patch_sim_indices)
    # print("sorted_max_vq_code_list:",sorted_max_vq_code_list)
    '''final_vqcode_list = []
    for code in sorted_max_vq_code_list:
        if not code in final_vqcode_list:
            final_vqcode_list.append(code)'''
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
    all_matched_vq_dict[cxr_id] = {"patches": sorted_max_raw_patches, "vq_code": sorted_max_vq_code_list}
    # print(all_matched_vq_dict[cxr_id])
    # exit(0)


# Store data (serialize)
with open(save_path, 'wb') as handle:
    pickle.dump(all_matched_vq_dict, handle)
print("Saved to :", save_path)
print("Finished")




'''
filepath = "top1000_patches_list.pickle"
with open(filepath, "rb") as f:
    data = pickle.load(f)


S = 3 # channel dim
W = 224 # width
H = 224 # height
batch_size = 10

# x = torch.randn(batch_size, S, H, W)
x = torch.randn(S, H, W)
# x = torch.randn(H, W)

size = 16 #32 # patch size
stride = 16 #32 # patch stride
patches = x.unfold(1, size, stride).unfold(2, size, stride) #.unfold(3, size, stride)
print(patches.shape)

'''