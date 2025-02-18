# Get entities for trainset
sh ./lanuch/get_entities_trainset.sh
# Get entities for testset
sh ./lanuch/get_entities_testset.sh

# Merge both entities from trainset and testset into one csv file
python3 merge_all_entities_to_csv.py

# Rank all the entities through their frequency
python3 get_rank.py

# END



# python3 get_keywords_from_report.py
# python3 clean_keywords_from_report.py