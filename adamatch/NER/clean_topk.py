import os
import pandas as pd
from tqdm import tqdm

# results_path = './outputs/mimic_top200_words.csv' #'./outputs/mimic_topk_words_clean2.csv'
results_path = './outputs_new/mimic_top500_words.csv' #'./outputs/mimic_topk_words_clean2.csv'
df = pd.read_csv(results_path)

# clean the same
out = df.groupby(['value'], group_keys=False).sum()
out = out.drop('Unnamed: 0', axis=1)
# a = df.groupby(['value'], group_keys=False).apply(lambda x: x).sum()

# out.to_csv('./outputs/mimic_top200_words_norepeat.csv')
out.to_csv('./outputs_new/mimic_top500_words_norepeat.csv')
print("Finished")