
# generate model
cd ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15
python3 zero_to_fp32.py ./ pytorch_model.bin

# we split the test set into several parts to test

CUDA_VISIBLE_DEVICES=0 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 0 

CUDA_VISIBLE_DEVICES=1 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 1 

CUDA_VISIBLE_DEVICES=2 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 2

CUDA_VISIBLE_DEVICES=3 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 3

CUDA_VISIBLE_DEVICES=0 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 4

CUDA_VISIBLE_DEVICES=1 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 5

CUDA_VISIBLE_DEVICES=2 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 6

CUDA_VISIBLE_DEVICES=3 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 7

CUDA_VISIBLE_DEVICES=0 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 8

CUDA_VISIBLE_DEVICES=0 python3 test_keywords_patches_viewPos.py \
--model_path ./checkpoints/adamatch_cyclic_keypatch_5_keywords_15 \
--results_path ./results/adamatch_cyclic_keypatch_5_keywords_15 --use-keywords True --key-topk 30 \
--keywords-path ./adamatch/data/topk_word_list/pickles/2023_09_21_19_00_21_newkeywords_top10.pickle \
--patches-path ./adamatch/data/topk_patch_list/2023_09_21_19_00_21_patches_matched_top20_patches_for_all_reports_dict.pickle \
--use-matched-patches True --keypatch-topk 5 --use-view-position True \
--split_total 10 --split_idx 9

