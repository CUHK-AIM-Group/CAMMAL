import os
import pandas as pd
from tqdm import tqdm
import pickle
from itertools import chain
results_path = './results_new'
results_list = os.listdir(results_path)
entity_group = ['Biological_structure', 'Biological_attribute',   'Detailed_description', 'Sign_symptom', 'Therapeutic_procedure', 'Disease_disorder']

save_keywords_pickle = './outputs_new/keywords_mimic.pickle'

output_dict = {}
bad_case = []
# th = 0.7
for filename in tqdm(results_list):
    path = os.path.join(results_path, filename)
    df = pd.read_csv(path)
    cxr_id = filename.split('.')[0]
    
    # new_df = df.loc[df['entity_group'].isin(entity_group)]
    # num = len(new_df)
    # max_len = 0
    # for entity in entity_group:
    #     out = df.loc[(df['entity_group'] == entity)]["value"].tolist()
    #     out_len = len(out)
    #     if out_len == 0:
    #         continue
    #     content[entity] = out
    #     if max_len < out_len:
    #         max_len = out_len
    try:
        entity_list0 = df.loc[(df['entity_group'] == entity_group[0])]["value"].tolist()
        entity_list1 = df.loc[(df['entity_group'] == entity_group[1])]["value"].tolist()
        entity_list2 = df.loc[(df['entity_group'] == entity_group[2])]["value"].tolist()
        entity_list3 = df.loc[(df['entity_group'] == entity_group[3])]["value"].tolist()
        entity_list4 = df.loc[(df['entity_group'] == entity_group[4])]["value"].tolist()
    except:
        bad_case.append(path)
        continue
    
    max_len = max(len(entity_list0),len(entity_list1),len(entity_list2),len(entity_list3),len(entity_list4))
    # for entity in entity_group:
    #     content[entity] += [0] * (max_len - len(content[entity]))
    entity_list0 += [''] * (max_len - len(entity_list0))
    entity_list1 += [''] * (max_len - len(entity_list1))
    entity_list2 += [''] * (max_len - len(entity_list2))
    entity_list3 += [''] * (max_len - len(entity_list3))
    entity_list4 += [''] * (max_len - len(entity_list4))

    data = list(chain.from_iterable(zip(entity_list0,entity_list1,entity_list2,entity_list3,entity_list4)))
    data = list(filter(None, data))
    # print(data)
    output_dict[cxr_id] = data
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
print("bad_case:",bad_case)
print("num bad:",len(bad_case))
with open(save_keywords_pickle, 'wb') as handle:
    pickle.dump(output_dict, handle)

print("Finished")


# import os
# import pandas as pd

# path = '7df738a1-ace27dbf-486c2c17-c90d032e-905567e9.csv'
# df = pd.read_csv(path)
# print(df)
# shuffled = df.sample(frac=1)
# print(shuffled)


# import pickle
# save_keywords_pickle = './outputs_new/keywords_mimic_clean.pickle'
# with open(save_keywords_pickle, "rb") as f:
#     data = pickle.load(f)
