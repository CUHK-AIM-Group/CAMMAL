cd ./adamatch/src/models/adamatch


# test on testset
## stage = 3
python3 mgca_module_test_dpt.py --gpus 8 --strategy ddp \
--img_encoder dpt_medium --dpt_return_stage 3 \
--batch_size 128 --emb_dim 256 \
--ckpt_path path_to_checkpoints_for_adamatch.ckpt


# test on trainset
## stage = 3
python3 mgca_module_test_dpt.py --gpus 8 --strategy ddp \
--img_encoder dpt_medium --dpt_return_stage 3 \
--batch_size 128 --emb_dim 256  --use_trainset True \
--ckpt_path path_to_checkpoints_for_adamatch.ckpt
