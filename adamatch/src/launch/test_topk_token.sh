cd ~code/img2text2img/MGCA/mgca/models/mgca

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module_test_token.py --gpus 4 --strategy ddp \
# --batch_size 128 --emb_dim 256 --split test \
# --ckpt_path ~code/img2text2img/MGCA/data/ckpts/MGCA/2023_07_29_21_39_42/epoch=15-step=7103.ckpt 

# two gpus
# CUDA_VISIBLE_DEVICES=0,1 python3 mgca_module_test_token.py --gpus 2 --strategy ddp \
# --batch_size 32 --emb_dim 256 --split train \
# --ckpt_path ~code/img2text2img/MGCA/data/ckpts/MGCA/2023_07_29_21_39_42/epoch=15-step=7103.ckpt 


python3 mgca_module_test_token.py --gpus 8 --strategy ddp \
--batch_size 2 --emb_dim 256 --split train --use_trainset True \
--ckpt_path ~code/img2text2img/MGCA/data/ckpts/MGCA/2023_08_14_07_26_14/last.ckpt

python3 mgca_module_test_token.py --gpus 8 --strategy ddp \
--batch_size 2 --emb_dim 256 --split test \
--ckpt_path ~code/img2text2img/MGCA/data/ckpts/MGCA/2023_08_14_07_26_14/last.ckpt


#~code/img2text2img/MGCA/data/ckpts/MGCA/2023_07_29_21_39_42/last.ckpt
