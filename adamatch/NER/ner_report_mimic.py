import nltk
from tqdm import tqdm
nltk.data.path.append("./")
# from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
from bio_recognizer_new import ner_all_prediction, ner_prediction, ner_init
import pickle
import os

# returns the predicted class along with the probability of the actual EnvBert model
# doc = ["The swan ganz catheter the left internal jugular vein catheter and the pacemaker leads are in correct and unchanged position. The effusion on the right has minimally decreased in extent but still occupies large parts of the right hemi thorax causing massive atelectasis at the right lung basis. On the left the heart border and the appearance of the lung parenchyma are unchanged",
# "No acute intrathoracic abnormality. The cardiomediastinal silhouette and pulmonary vasculature are unremarkable. There is no pleural effusion or pneumothorax. No definite focal consolidation is identified."]
import argparse

parser = argparse.ArgumentParser(description='Show args')
parser.add_argument('--split_idx', type=int, help=0)
parser.add_argument('--gpu_id', type=int, help=0)
parser.add_argument('--split_num', type=int, help=20)
parser.add_argument('--phase', type=str, default='test')

args = parser.parse_args()

split_idx = args.split_idx
split_num = args.split_num
gpu_id = args.gpu_id

pickle_path = '../../materials/mimic_xray_'+args.phase+'_caption.pickle'
# pickle_path = '~dataset/x_ray/mimic_xray/info/mimic_xray_train_caption.pickle'
#'~dataset/x_ray/mimic_xray/anno_mgca_new/captions.pickle'
with open(pickle_path, "rb") as f:
    path2sent = pickle.load(f)

total = len(path2sent)
each_num = total // split_num
start_pos = max(0, split_idx * each_num)
end_pos = min(total, (split_idx+1)*each_num)
print("split_idx:",split_idx,"split_num:",split_num)
print("start_pos:",start_pos,"end_pos:",end_pos)
print("total:",total)
key_list = list(path2sent.keys())
key_list.sort()

part_keys_list = key_list[start_pos:end_pos]

pipe = ner_init(compute='gpu', gpu_id=gpu_id)
# for i, path in enumerate(tqdm(path2sent)):
for path in tqdm(part_keys_list):
    cxr_id = path.split('/')[-1].split('.')[0]
    report = path2sent[path]
    # print(cxr_id,report)
    # report = '. '.join(report)
    out = ner_prediction(corpus=report, compute='gpu', pipe=pipe)
    out.to_csv(os.path.join('./results_new', cxr_id + '.csv'))
    # break
    # returns a dataframe output
    # out = ner_all_prediction(corpus_list=doc, compute='gpu') #pass compute='gpu' if using gpu
    # print(out)
    # print(type(out))
    # out[0].to_csv('test1.csv')
    # out[1].to_csv('test2.csv')
