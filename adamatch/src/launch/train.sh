# CUDA_VISIBLE_DEVICES=0,1 python3 ./mgca/models/mgca/mgca_module.py --gpus 2 --strategy ddp

cd ~code/img2text2img/MGCA/mgca/models/mgca

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module.py --gpus 4 --strategy ddp \
--emb_dim 256 --learning_rate 6e-3 --weight_decay 3e-2 --batch_size 128 --max_epochs 30  

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module.py --gpus 4 --strategy ddp \
--emb_dim 256 --learning_rate 6e-3 --weight_decay 3e-2 --batch_size 128 --max_epochs 60  


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module.py --gpus 4 --strategy ddp \
--emb_dim 256 --learning_rate 6e-3 --weight_decay 3e-2 --batch_size 124 --max_epochs 30 


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module.py --gpus 4 --strategy ddp \
--emb_dim 256  --batch_size 128 #168 
#--learning_rate 6e-3 --weight_decay 3e-2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mgca_module.py --gpus 4 --strategy ddp \
# --emb_dim 256 --learning_rate 6e-3 --weight_decay 3e-2 --batch_size 168 #128 #224 #256 #144