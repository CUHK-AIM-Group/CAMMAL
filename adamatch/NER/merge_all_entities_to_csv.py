import os
import pandas as pd
from tqdm import tqdm
results_path = './results_new' #'./results'
results_list = os.listdir(results_path)

output = []
# th = 0.7
for filename in tqdm(results_list):
    path = os.path.join(results_path, filename)
    df = pd.read_csv(path)
    output.append(df)
    # df to dict
    # for index, row in df.iterrows():
    #     entity_name = row['entity_group']
    #     item_name = row['value']
    #     score = row['score']
    #     if score < th: 
    #         continue
    #     if not entity_name in output:
    #         output[entity_name] = {}
    #     if not item_name in output[entity_name]:
    #         output[entity_name][item_name] = 1
    #     else:
    #         output[entity_name][item_name] += 1

out = pd.concat(output)
out.to_csv('./outputs_new/mimic_all.csv')
print("Finished")

# import pandas as pd

# path = '7df738a1-ace27dbf-486c2c17-c90d032e-905567e9.csv'
# df = pd.read_csv(path)

# df0 = pd.read_csv(datalist[0])
# df1 = pd.read_csv(datalist[1])

# out = pd.concat([df0, df1])
# entity_group = list(set(out.loc[:,"entity_group"].tolist()))
# value = list(set(out.loc[:,"value"].tolist()))

# for entity in entity_group:
#     for val in value:
#         res = out.loc[(out['entity_group'] == entity) & (out['value'] == val) & (out['score'] >= th)]

#         print(entity, val, out.loc[entity][val]['score'] >th)
#         break