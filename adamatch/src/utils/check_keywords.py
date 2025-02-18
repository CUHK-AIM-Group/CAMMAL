import os
import pickle
import _pickle as cPickle

from tqdm import tqdm
import pandas as pd
import numpy as np

root = '~code/img2text2img/MGCA/data/topk_word_list/pickles/2023_07_29_21_39_42_keywords_top10.pickle'
caption_root = '~dataset/x_ray/mimic_xray/anno_mgca_new/captions.pickle'
with open(root, 'rb') as file:
    keyword_dict = cPickle.load(file)

with open(caption_root, 'rb') as file:
    caption_dict = cPickle.load(file)

# keys_list_0 = keyword_dict.keys()
keys_list_1 = caption_dict.keys()

# keys_list_0 = list(keys_list_0)
# keys_list_1 = list(keys_list_1)

# keys_list_1 = [item.split('/')[-1].split('.')[0] for item in keys_list_1]

# keys_list_0.sort()
# keys_list_1.sort()

# print("keywords:", len(keys_list_0))
# print("captions:", len(keys_list_1))
# out = keys_list_0==keys_list_1
# print(out)

topk = 25
topk_list = [30 + (i+1)*5 for i in range(20)] #[5, 10, 15, 20, 25, 30]
results = []
for topk in topk_list:
    matched_count_list = []
    for i, key in tqdm(enumerate(keys_list_1), total=len(keys_list_1)):
        # key = keys_list_1[i]
        captions = caption_dict[key]
        key2 = key.split('/')[-1].split('.')[0]
        max_len = min(len(keyword_dict[key2]), topk)
        keywords = keyword_dict[key2][:max_len]
        new_kewords = []
        for words in keywords:
            new_kewords.extend(words.split(' '))
        new_kewords = list(set(new_kewords))
        # print("i=",i)
        # print("cxr_id:", key)
        # print("Keywords:", keywords)
        # print("Captions:", captions)
        # if i > 20:
        #     break
        sentence = ' '.join(captions)
        count = 0
        # exist_words = []
        for w in new_kewords:
            if w in sentence:
                # exist_words.append(w)
                count += 1
        matched_count_list.append(count)
        # print("Keywords:", exist_words)
        # print("Captions:", sentence)
        # if i >10:
        #     break

    matched_count_list = np.array(matched_count_list)
    # results.append(matched_count_list)
    print("topk:",topk, "Mean:", matched_count_list.mean(),"Total:",matched_count_list.sum())

# out = (results[0] == results[1]).all()
# print("equal 0=1:", out)

''''
topk: 5 Mean: 2.7361097956829084 Total: 635560
topk: 10 Mean: 4.334600449445942 Total: 1006867
topk: 15 Mean: 5.473747879768905 Total: 1271475
topk: 20 Mean: 6.424338961452692 Total: 1492284
topk: 25 Mean: 7.2466011726922845 Total: 1683284
topk: 30 Mean: 7.965770644808555 Total: 1850337
topk: 35 Mean: 8.599256950483456 Total: 1997487
topk: 40 Mean: 9.166195982538767 Total: 2129179
topk: 45 Mean: 9.683670130787046 Total: 2249381
topk: 50 Mean: 10.154852208053864 Total: 2358830
topk: 55 Mean: 10.59304478100273 Total: 2460616
topk: 60 Mean: 10.997128539817295 Total: 2554479
topk: 65 Mean: 11.37450814943647 Total: 2642139
topk: 70 Mean: 11.72248435118776 Total: 2722969
topk: 75 Mean: 12.045323437486546 Total: 2797960
topk: 80 Mean: 12.344002651903256 Total: 2867339
topk: 85 Mean: 12.621191117846104 Total: 2931726
topk: 90 Mean: 12.874211101831364 Total: 2990499
topk: 95 Mean: 13.104728653470291 Total: 3044045
topk: 100 Mean: 13.30579974686378 Total: 3090751
topk: 105 Mean: 13.482474191298659 Total: 3131790
topk: 110 Mean: 13.630799962115669 Total: 3166244
topk: 115 Mean: 13.749864391310712 Total: 3193901
topk: 120 Mean: 13.844433155678775 Total: 3215868
topk: 125 Mean: 13.917511171572976 Total: 3232843
topk: 130 Mean: 13.97029523948925 Total: 3245104
topk: max Mean: 14.071726234039073 Total: 3268665
''''