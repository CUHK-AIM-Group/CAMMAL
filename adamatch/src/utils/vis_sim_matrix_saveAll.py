# import pickle
# import torch
# import torch.nn.functional as F
# out_dir = '~code/img2text2img/MGCA/data/tmp'
# filepath="~code/img2text2img/MGCA/data/results/2023_07_29_21_39_42/cuda:0_idx_0.pickle"
# with open(filepath, "rb") as f:
#     data = pickle.load(f)

# data[0]['sim']

# a = F.softmax(path2sent[0]['sim'])


import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from transformers import BertTokenizer
import cv2 
import os
from tqdm import tqdm

out_dir = '~code/img2text2img/MGCA/data/tmp'

file_root = "~code/img2text2img/MGCA/data/results/2023_07_29_21_39_42_test"
file_list = os.listdir(file_root)

for pickle_filename in tqdm(file_list):
    if not 'pickle' in pickle_filename:
        continue
    filepath = os.path.join(file_root, pickle_filename)
    print(filepath)
    # filepath="~code/img2text2img/MGCA/data/results/2023_07_29_21_39_42/cuda:0_idx_0.pickle"
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    folder_name = filepath.split('/')[-2] # xxx_xx_xx
    save_folder = os.path.join(out_dir, folder_name)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)#, exist_ok=True)
    # data[0]['sim']
    # 'path': img_path,
    # 'sentence': sentence,
    # 'input_ids': input_ids.cpu().detach().numpy(),
    # 'special_token_mask': special_token_mask.cpu().detach().numpy(),
    # 'sim'

    tokenizer = BertTokenizer.from_pretrained("~code/img2text2img/Bio_ClinicalBERT") # ("./Bio_ClinicalBERT")#

    # n_idx = 10
    for n_idx in tqdm(range(len(data))):
        sample = data[n_idx]
        # print(sample['path'])
        # print(sample['sentence'])
        # print("sample['input_ids']:", len(sample['input_ids']))
        # print("sample['special_token_mask']:", len(sample['special_token_mask']))

        img_path = sample['path'] #'69edea97-d76e1e86-638a39dc-13ee8420-6f3385ef.jpg'
        xray_img = cv2.imread(img_path)

        filename = img_path.split('/')[-1].split('.')[0]
        save_pdf_path = os.path.join(save_folder, filename + '.pdf')
        save_txt_path = os.path.join(save_folder, filename + '.txt')
        txt_file = open(save_txt_path, 'w')

        decode_list = [tokenizer.decode(ids).replace(' ','') for ids in sample['input_ids']]
        decode_list = decode_list[1:]
        # decode_out = tokenizer.decode(sample['input_ids'])[0]
        # print("decode_list:",len(decode_list), decode_list)
        # a = F.softmax(torch.tensor(sample['sim']))


        txt_file.write("Path: {}\n\n".format(sample['path']))
        txt_file.write("Sentence: {}\n\n".format(sample['sentence']))
        txt_file.write("Decode_list: {}\n\n".format(', '.join(decode_list)))
        txt_file.write("Input ids: {}\n\n".format(sample['input_ids']))
        txt_file.write("Special token masks: {}\n".format(sample['special_token_mask']))

        a = torch.tensor(sample['sim'])

        dim1, dim2 = a.shape
        num_patch = int(math.sqrt(dim1))
        a = torch.reshape(a, [num_patch, num_patch, dim2])
        # sort
        topk=4
        sorted_a, indices = torch.sort(a, dim=2, descending=True)
        sorted_a = sorted_a[:,:,:topk] # top5
        indices = indices[:,:,:topk]

        tmp = np.zeros([num_patch, num_patch])
        plt.figure(figsize=(15,15))
        plt.plot(tmp)
        plt.grid()

        x = np.array([i for i in range(num_patch)])
        y = np.array([0 for i in range(num_patch)])

        plt.title("RUNOOB grid() Test")
        plt.xlabel("x - label")
        plt.ylabel("y - label")

        plt.xlim(0,num_patch + 2)
        plt.ylim(0,num_patch + 2)

        plt.xticks(np.arange(num_patch+2))#,1))
        plt.yticks(np.arange(num_patch+2))#,1))

        plt.imshow(xray_img, extent=[1, 15, 1, 15])
        # plt.figure(figsize=(15,15))
        # plt.plot(x, y, linewidth=0)

        # draw idx image
        token_img = []
        for i in range(num_patch):
            token_img.append([])
            for j in range(num_patch):
                # print("indices[i][j]:",indices[i][j].shape)
                each_indice = indices[i][j].numpy().tolist()[:topk]
                word_list = [decode_list[idx] for idx in each_indice]
                if '[ S E P ]' in word_list:
                    word_list.remove('[ S E P ]')
                # print("indices[i][j]:",each_indice)
                # word_list = decode_list[each_indice]
                token_img[i].append(word_list)
                # print(j, i, ','.join(word_list))
                # plt.text(j, i, ','.join(word_list))

        for i in range(num_patch):
            y = num_patch - i
            for j in range(num_patch):
                x = num_patch - j
                word_list = token_img[i][j]
                plt.text(x, y, '\n'.join(word_list).replace(' ',''),color='orange')

        # print("token_img:",len(token_img),len(token_img[0]))


        '''img = img.numpy()

        plt.xticks(np.arange(len(decode_list)), decode_list) 

        # plt.imshow(img[:,:,0], cmap='gray')
        plt.imshow(img)#, cmap='gray')'''

        plt.savefig(save_pdf_path, format="pdf")#, bbox_inches="tight")
        plt.show()
        plt.close()
        txt_file.close()
        # break
