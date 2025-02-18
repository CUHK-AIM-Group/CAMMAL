import pickle
from tqdm import tqdm

src_keywords_pickle = './outputs_new/keywords_mimic.pickle'
save_keywords_pickle = './outputs_new/keywords_mimic_clean.pickle'

with open(src_keywords_pickle, "rb") as f:
    data = pickle.load(f)

new_data = {}
for key in tqdm(data):
    item = data[key]
    # print(key, item)
    words_list = []
    for val in item: # each keyword
        words = val.split(' ')
        new_words = []
        for w in words: # each word
            if not w in new_words:
                new_words.append(w)
        out_words = ' '.join(new_words)
        words_list.append(out_words)
    new_data[key] = words_list
    # print(new_data)
    # exit(0)
# print(new_data)


with open(save_keywords_pickle, 'wb') as handle:
    pickle.dump(new_data, handle)
