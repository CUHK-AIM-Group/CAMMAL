
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"


#!/bin/bash
export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='adamatch_cyclic_keypatch_5_keywords_15'
export checkpoint_dir_name="${model_name}__${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_training_root='./checkpoints'
export local_output_dir="${local_training_root}/${checkpoint_dir_name}"
export dbfs_output_dir=''
export tensorboard_display_dir="${local_output_dir}/runs"
export input_model="databricks/dolly-v2-3b"
# export input_model="./pretrained_model/dolly-v2-3b"

# echo $input_model 
# echo $deepspeed_config
# echo $local_output_dir 
# # 4

deepspeed --num_gpus=8 \
     --module training.trainer_findings \
     --input-model $input_model \
     --deepspeed $deepspeed_config \
     --epochs 5 \
     --local-output-dir $local_output_dir \
     --dbfs-output-dir "" \
     --per-device-train-batch-size 24 \
     --per-device-eval-batch-size 24 \
     --logging-steps 50 \
     --save-total-limit 2 \
     --eval-steps 1500 \
     --warmup-steps 50 \
     --test-size 200 \
     --lr 5e-6 \
     --stage 1 \
     --only-use-findings False \
     --use-origin-report True \
     --use-keywords True \
     --keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
     --key-topk 15 \
     --use-matched-patches True \
     --keypatch-topk 5 \
     --patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
     --use-view-position True 

