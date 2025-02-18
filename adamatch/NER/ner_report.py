# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForTokenClassification

# tokenizer = AutoTokenizer.from_pretrained("~code/img2text2img/biomedical-ner-all")
# model = AutoModelForTokenClassification.from_pretrained("~code/img2text2img/biomedical-ner-all")

# pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu
# out = pipe("""The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.""")
# print(out)
import pandas as pd

import nltk
nltk.data.path.append("./")
# from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
from bio_recognizer_new import ner_all_prediction, ner_prediction, ner_init
# returns the predicted class along with the probability of the actual EnvBert model
# doc = ["The swan ganz catheter the left internal jugular vein catheter and the pacemaker leads are in correct and unchanged position. The effusion on the right has minimally decreased in extent but still occupies large parts of the right hemi thorax causing massive atelectasis at the right lung basis. On the left the heart border and the appearance of the lung parenchyma are unchanged",
# "No acute intrathoracic abnormality. The cardiomediastinal silhouette and pulmonary vasculature are unremarkable. There is no pleural effusion or pneumothorax. No definite focal consolidation is identified."]


import _pickle as cPickle
path = './material/anno/mimic_xray_test_caption.pickle'
with open(path, "rb") as f:
    caption_dict = cPickle.load(f)
doc = []

for idx, key in enumerate(caption_dict):
    if idx >= 1:#20:
        break
    print(idx, key, caption_dict[key])
    doc.append(caption_dict[key])

# doc = ["Lateral view somewhat limited due to overlying motion artifact. The lungs are low in volume.  There is no focal airspace consolidation to suggest pneumonia.  A 1.2-cm calcified granuloma just below the medial aspect of the right hemidiaphragm is unchanged from prior study.  No pleural effusions or pulmonary edema. There is no pneumothorax.  The inferior sternotomy wire is fractured but unchanged. Surgical clips and vascular markers in the thorax are related to prior CABG surgery. No evidence of acute cardiopulmonary process.", "No acute intrathoracic abnormality. The cardiomediastinal silhouette and pulmonary vasculature are unremarkable. There is no pleural effusion or pneumothorax. No definite focal consolidation is identified."]
# doc = doc[0]
# """
# 	CASE: A 28-year-old previously healthy man presented with a 6-week history of palpitations. 
#       The symptoms occurred during rest, 2–3 times per week, lasted up to 30 minutes at a time 
#       and were associated with dyspnea. Except for a grade 2/6 holosystolic tricuspid regurgitation 
#       murmur (best heard at the left sternal border with inspiratory accentuation), physical 
#       examination yielded unremarkable findings.
#       """
# pipe = ner_init(compute='gpu', gpu_id=0)

# returns a dataframe output
out = ner_all_prediction(corpus_list=doc, compute='gpu', gpu_id=0) #pass compute='gpu' if using gpu
# print(out)
# print(type(out))
# out[0].to_csv('test1.csv')
# out[1].to_csv('test2.csv')
df = pd.concat(out)
df.to_csv('test.csv')
