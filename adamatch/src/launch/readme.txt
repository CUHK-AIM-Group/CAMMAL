# get matched keypatches

## 1. get the similarity matrices for image-text pair (both trainset and testset)

sh ./launch/test_dpt.sh 

## 2. merge similarity matrices into a pickle file

python3 mgca/utils/merge_topk_patches.py

## 3. get topk raw patches and vq code
    # for mimic-cxr dataset
    python3 mgca/utils/get_topk_raw_patches_dpt.py
    # for openI dataset
    python3 mgca/utils/get_topk_raw_patches_dpt_openI.py

## 4. match topk keypatches for reports from trainset and testset

sh ./launch/test_topk_patch_token_dpt.sh

------------------------------------------------------------------------------------------
# extract keywords from real reports 

## 0. Please refer to this file : ~code/img2text2img/NER_openI/readme.txt

# get matched keywords

## 1. match topk keywords with images for trainset and testset

sh ./launch/test_topk_token_dpt.sh

## 2. merge matched keywords into one pickle file

sh ./launch/get_topk_keywords.sh