cd ./adamatch/src

# two gpus
# trainset
CUDA_VISIBLE_DEVICES=0,1 python3 mgca_module_test_patch_token_dpt.py --gpus 2 --strategy ddp \
--batch_size 512 --emb_dim 256 --split train --use_trainset True \
--ckpt_path ./adamatch/data/ckpts/2023_08_14_07_26_14/last.ckpt

# testset
CUDA_VISIBLE_DEVICES=0,1 python3 mgca_module_test_patch_token_dpt.py --gpus 2 --strategy ddp \
--batch_size 512 --emb_dim 256 --split test \
--ckpt_path ./adamatch/data/ckpts/2023_08_14_07_26_14/last.ckpt

# merge all the topk patches for each report 
cd ./adamatch/src/utils
python3 merge_topk_matched_patches.py
# saved to 
#  ./adamatch/data/topk_patch_list/pickles/2023_08_14_07_26_14_patches_matched_top20_patches_for_all_reports.pickle