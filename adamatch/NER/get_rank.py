import os
import pandas as pd
from tqdm import tqdm

th= 0.7
results_path = './outputs_new/mimic_all.csv'
df = pd.read_csv(results_path)

# entity_group = list(set(df.loc[:,"entity_group"].tolist()))
# entity_group = ['Outcome', 'Clinical_event', 'Biological_structure', 'Biological_attribute',   'Detailed_description', 'Sign_symptom', 'Quantitative_concept', 'Nonbiological_location', 'Area',  'Therapeutic_procedure', 'Disease_disorder']

# out = df.loc[df['entity_group'].isin(entity_group)]
# out = out.loc[out['score'] >= th]
# value = list(set(out.loc[:,"value"].tolist()))
# print(entity_group)
# entity_list = []
# entity_val = []
# count_list = []


# for entity in tqdm(entity_group):
#     print("entity:",entity)
#     for val in tqdm(value):
#         res0 = out.loc[(out['entity_group'] == entity) & (out['value'] == val)]
#         if len(res0) == 0 :
#             continue
#         entity_list.append(entity)
#         entity_val.append(val)
#         count_list.append(len(res))
#         # print(entity, val, out.loc[entity][val]['score'] >th)
#         # break

# save_tab = pd.DataFrame({
#     "entity_group": entity_list,
#     "value": entity_val,
#     "score": count_list
# })
# save_tab.to_csv('statistics.csv')
# print("Finished")


# entity_group = ['Outcome', 'Clinical_event', 'Biological_structure', 'Biological_attribute',   'Detailed_description', 'Sign_symptom', 'Quantitative_concept', 'Nonbiological_location', 'Area',  'Therapeutic_procedure', 'Disease_disorder']
entity_group = ['Biological_structure', 'Biological_attribute',   'Detailed_description', 'Sign_symptom', 'Therapeutic_procedure', 'Disease_disorder']
th= 0.7
out = df.loc[df['entity_group'].isin(entity_group)]
out = out.loc[out['score'] >= th]
out['score'] = 1
out = out.drop('Unnamed: 0.1', axis=1)
res = out.drop('Unnamed: 0', axis=1)
tab = res.groupby(['entity_group', 'value']).sum()#.apply(lambda x: x)
# tab['rank'] = tab['score'].rank(ascending=False)
# entity_group = list(set(out.loc["entity_group"].tolist()))
from tqdm import tqdm
new_tab = {
    'entity_group' : [],
    'value' : [],
    'score' : []
}
for index, row in tqdm(tab.iterrows(), total=len(tab)):
    new_tab['entity_group'].append(row.name[0])
    new_tab['value'].append(row.name[1])
    new_tab['score'].append(row['score'])

new_df = pd.DataFrame(new_tab)
entity_group = list(set(new_df["entity_group"].tolist()))
value = list(set(new_df["value"].tolist()))

entity_topk = []
topk=500
for entity in tqdm(entity_group):
    print("entity:",entity)
    # entity = 'Therapeutic_procedure'
    tmp = new_df.loc[(new_df['entity_group'] == entity)].sort_values(by=['score'], ascending=False)
    entity_topk.append(tmp.iloc[:topk])

final_out = pd.concat(entity_topk)
final_out.to_csv('./outputs_new/mimic_top'+str(topk)+'_words.csv')
print("Finished")

# Quantitative_concept
# Outcome
# Area
# Nonbiological_location
# 'Clinical_event', 