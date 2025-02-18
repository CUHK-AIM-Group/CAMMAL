cd ./AdaMatch-Cyclic/adamatch/src/models/adamatch


python3 mgca_module_test_token_dpt.py --gpus 8 --strategy ddp \
--batch_size 2 --emb_dim 256 --split train --use_trainset True \
--img_encoder dpt_medium --dpt_return_stage 3 \
--ckpt_path path_to_checkpoints_for_adamatch.ckpt

python3 mgca_module_test_token_dpt.py --gpus 8 --strategy ddp \
--batch_size 2 --emb_dim 256 --split test \
--img_encoder dpt_medium --dpt_return_stage 3 \
--ckpt_path path_to_checkpoints_for_adamatch.ckpt
