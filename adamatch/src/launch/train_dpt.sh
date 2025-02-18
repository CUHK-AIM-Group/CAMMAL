cd ~/code/image2text2image/FLIP_medical/mgca/models/mgca

# dpt_return_stage = 3
python3 mgca_module.py --gpus 8 --strategy ddp \
--emb_dim 256 --learning_rate 6e-3 --weight_decay 3e-2 --batch_size 112 --max_epochs 15 \
--img_encoder dpt_medium --dpt_return_stage 3
