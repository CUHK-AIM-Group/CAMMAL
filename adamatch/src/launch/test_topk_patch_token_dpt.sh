cd ./adamatch/src/

# two gpus
# trainset
CUDA_VISIBLE_DEVICES=0,1 python3 mgca_module_test_patch_token_dpt.py --gpus 2 --strategy ddp \
--batch_size 512 --emb_dim 256 --split train --use_trainset True \
--img_encoder dpt_medium --dpt_return_stage 3 \
--ckpt_path path_to_adamatch_checkpoint.ckpt

# testset
CUDA_VISIBLE_DEVICES=0,1 python3 mgca_module_test_patch_token_dpt.py --gpus 2 --strategy ddp \
--batch_size 512 --emb_dim 256 --split test \
--img_encoder dpt_medium --dpt_return_stage 3 \
--ckpt_path path_to_adamatch_checkpoint.ckpt
