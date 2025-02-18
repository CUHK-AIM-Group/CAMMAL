# python3 save_topk_keywords.py 
cd ./adamatch/src/utils

# trainset 
python3 save_topk_keywords_split.py --split_num 10 --split_idx 0
python3 save_topk_keywords_split.py --split_num 10 --split_idx 1
python3 save_topk_keywords_split.py --split_num 10 --split_idx 2
python3 save_topk_keywords_split.py --split_num 10 --split_idx 3
python3 save_topk_keywords_split.py --split_num 10 --split_idx 4
python3 save_topk_keywords_split.py --split_num 10 --split_idx 5
python3 save_topk_keywords_split.py --split_num 10 --split_idx 6
python3 save_topk_keywords_split.py --split_num 10 --split_idx 7
python3 save_topk_keywords_split.py --split_num 10 --split_idx 8
python3 save_topk_keywords_split.py --split_num 10 --split_idx 9
python3 save_topk_keywords_split.py --split_num 10 --split_idx 10


# testset
python3 save_topk_keywords.py 
# python3 merge_topk_keywords.py


# all
python3 merge_topk_keywords.py