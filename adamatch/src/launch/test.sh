# CUDA_VISIBLE_DEVICES=0,1 python3 ./mgca/models/mgca/mgca_module.py --gpus 2 --strategy ddp

cd ~code/img2text2img/MGCA/mgca/models/mgca

# CUDA_VISIBLE_DEVICES=0,1 python3 mgca_module_test.py --gpus 2 --strategy ddp \
# --batch_size 72 --ckpt_path ~code/img2text2img/MGCA/data/ckpts/MGCA/2023_07_29_21_39_42/last.ckpt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module_test.py --gpus 4 --strategy ddp \
# --batch_size 128 --emb_dim 256 --ckpt_path ~code/img2text2img/MGCA/data/ckpts/MGCA/2023_07_29_21_39_42/epoch=15-step=7103.ckpt
#~code/img2text2img/MGCA/data/ckpts/MGCA/2023_07_29_21_39_42/last.ckpt

cd ~/code/image2text2image/FLIP_medical/mgca/models/mgca

# test on testset
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module_test.py --gpus 4 --strategy ddp \
--batch_size 128 --emb_dim 256 --ckpt_path ~/code/image2text2image/FLIP_medical/data/ckpts/MGCA/2023_08_14_07_26_14/last.ckpt


# test on trainset
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module_test.py --gpus 4 --strategy ddp \
# --batch_size 128 --emb_dim 256 --ckpt_path ~code/img2text2img/MGCA/data/ckpts/MGCA/2023_08_14_07_26_14/last.ckpt --use_trainset True

