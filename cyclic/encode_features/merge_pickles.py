import pickle
import os

split_num= 6 
root = '~dataset/x_ray/openI_data/data'
save_pickle = os.path.join(root, 'openI_vqgan1024_codebook_all.pickle')
data_dict = {}
for split_id in range(split_num):
    path = os.path.join(root, 'openI_vqgan1024_codebook_'+ str(split_id) +'_'+ str(split_num) +'.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)
        # print(type(data))
        # print(data)
        print(len(data.keys()))
        data_dict.update(data)

print(len(data_dict.keys()))

with open(save_pickle, 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
