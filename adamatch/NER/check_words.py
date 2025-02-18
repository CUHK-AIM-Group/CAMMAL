import os
import pandas as pd
from tqdm import tqdm

# results_path = './outputs/mimic_top200_words_norepeat.csv' #'./outputs/mimic_topk_words_norepeat.csv'
results_path = './outputs_new/mimic_top500_words_norepeat.csv' #'./outputs/mimic_topk_words_norepeat.csv'
df = pd.read_csv(results_path)

# results_path = './outputs/mimic_top200_words.csv' #'./outputs/mimic_topk_words_clean2.csv'
results_path = './outputs_new/mimic_top500_words.csv' #'./outputs/mimic_topk_words_clean2.csv'
df2 = pd.read_csv(results_path)

entity_group = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    # print(row['value'], df2.loc[df2['value'] == row['value']].iloc[0]['entity_group'])
    entity_group.append(df2.loc[df2['value'] == row['value']].iloc[0]['entity_group'])
    # row['value']
    # row['score']

df['entity_group'] = entity_group

# save_path = './outputs/mimic_top200_words_norepeat_with_entity.csv'
save_path = './outputs_new/mimic_top500_words_norepeat_with_entity.csv'
df.to_csv(save_path)



