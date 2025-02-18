from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np

from pathlib import Path
import _pickle as cPickle
import random
from typing import List 

from .consts import RESPONSE_KEY_NL
import pandas as pd
import re
import os
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

CXR_VQ_OUTPUT_INSTRUCTION_LIST = [
    "Generate a chest X-ray image that corresponds to the entered free-text radiology reports for the chest X-ray image.",
    "Use the free-text radiology reports for the chest X-ray image to produce a corresponding chest X-ray image.",
    "Utilize the entered free-text radiology reports for the chest X-ray image to create a corresponding chest X-ray image.",
    "Create a chest X-ray image that matches the free-text radiology reports entered for the chest X-ray image.",
    "Produce a chest X-ray image that is consistent with the free-text radiology reports entered for the chest X-ray image.",
    "Based on the free-text radiology reports for the chest X-ray image, generate a corresponding chest X-ray image.",
    "Use the free-text radiology reports entered for the chest X-ray image to create a corresponding chest X-ray image.",
    "Generate a chest X-ray image that is in accordance with the free-text radiology reports for the chest X-ray image entered.",
    "Create a chest X-ray image that corresponds to the free-text radiology reports entered for the chest X-ray image.",
    "Utilize the entered free-text radiology reports for the chest X-ray image to produce a corresponding chest X-ray image."
]

CXR_VQ_INPUT_INSTRUCTION_LIST = [
    "Generate free-text radiology reports for the entered chest X-ray images.",
    "Use the entered chest X-ray images to create corresponding free-text radiology reports.",
    "Based on the entered chest X-ray images, produce free-text radiology reports.",
    "Create free-text radiology reports that correspond to the entered chest X-ray images.",
    "Utilize the entered chest X-ray images to generate corresponding free-text radiology reports.",
    "Generate free-text radiology reports in accordance with the entered chest X-ray images.",
    "Use the entered chest X-ray images to create accurate free-text radiology reports.",
    "Produce free-text radiology reports that match the entered chest X-ray images.",
    "Create free-text radiology reports that are consistent with the entered chest X-ray images.",
    "Utilize the entered chest X-ray images to generate comprehensive free-text radiology reports."
]

CXR_VQ_VQ_REPLACE_TEMPLATE = "<image>"

CXR_VQ_CODE_BOOK_SIZE = 1024
CXR_VQ_VQ_LEN = 256

CXR_VQ_TOKENIZER_LEN = 50281

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

class MimicCxrVqDataset(Dataset): 
    def __init__(self, report_path: str, vq_path: str, tokenizer_len: int, mode: str, stage: int, split_path: str, split_idx: int, split_total: int, \
                    use_keywords: bool, keywords_path: str, key_topk: int, use_matched_patches: bool, patches_path: str, keypatch_topk: int, use_it: bool,
                        use_view_position: bool):
        assert tokenizer_len == CXR_VQ_TOKENIZER_LEN
        assert stage in [1, 2]

        parse_fun_map = {1: self._parse_report_fi, 2: self._parse_report_i}
        
        self.split_path = Path(split_path)
        self.report_path = Path(report_path)
        self.db = pd.read_csv(self.split_path / "mimic-cxr-2.0.0-split.csv", index_col="dicom_id", dtype=str)
        
        if mode == "train":
            self.db = self.db.loc[self.db['split'] == 'train']
        elif mode == "test":
            self.db = self.db.loc[(self.db['split'] == 'test') | (self.db['split'] == 'validate')]
        else:
            raise ValueError(mode)

        with open(vq_path, "rb") as f:
            self.cxr_vq = cPickle.load(f)
        error_dicom_ids = self._check_vq()
        self.dicom_ids = sorted(list(set(self.db.index) - error_dicom_ids))

        # load keywords
        self.keywords_dict = None
        if use_keywords:
            with open(keywords_path, "rb") as f:
                self.keywords_dict = cPickle.load(f)
        
        # load patches
        self.patches_dict = None
        if use_matched_patches:
            with open(patches_path, "rb") as f:
                self.patches_dict = cPickle.load(f)
        
        # load meta data
        meta_data = None
        if use_view_position:
            meta_data = pd.read_csv(self.split_path / "mimic-cxr-2.0.0-metadata.csv", index_col='dicom_id', dtype=str)
            meta_data = meta_data[['ViewPosition']]
            meta_data = meta_data.to_dict()['ViewPosition']
            view_pos_dict = {'LATERAL': 'lateral', 'PA': 'PA' , 'AP': 'AP', 'LL': 'left lateral'}
        
        # load captions
        self.caption_dict = None
        # if use_origin_report:
        caption_path = os.path.join(split_path, "mimic_xray_"+mode+"_caption.pickle")
        with open(caption_path, "rb") as f:
            self.caption_dict = cPickle.load(f)
        
        # with open(self.split_path / "mimic-cxr-2.0.0-selected-pa-ap-earlist-study.pickle", "rb") as f:
        #     self.selected_dicom_ids = set(cPickle.load(f).keys())

        split_file = "mimic-cxr-2.0.0-"+ mode +".csv"
        df_split = pd.read_csv(self.split_path / split_file)
        self.selected_dicom_ids = df_split['dicom_id'].tolist()
        self.df_split = df_split

        self.outputs = [] 
        # self.tokenizer_len = tokenizer_len


        total_num = len(self.selected_dicom_ids) #len(self.outputs)
        num_each_split = int(np.ceil(total_num / split_total) )
        start_idx = split_idx * num_each_split
        end_idx = min((split_idx + 1 ) * num_each_split, total_num)
        if split_idx == (split_total-1):
            end_idx =  total_num
        print("total_num:",total_num)
        print("start_idx:",start_idx,"end_idx:",end_idx)
        self.selected_dicom_ids = self.selected_dicom_ids[start_idx:end_idx]
        print("self.selected_dicom_ids:", len(self.selected_dicom_ids))

        for i, dicom_id in enumerate(tqdm(self.selected_dicom_ids, colour="green", unit='pair')):
            # if (stage == 1) or (dicom_id in self.selected_dicom_ids):
            # if (dicom_id in self.selected_dicom_ids):
            try:
                io_type = "input" if i % 2 == 0 else "output"
                cxr_vq = self.cxr_vq[dicom_id]
                cxr_vq_shifted = [x + tokenizer_len for x in cxr_vq]
                # report = self._load_report(dicom_id, parse_fun_map[stage])
                info = df_split.loc[(df_split['dicom_id'] == dicom_id),['finding','study_id', 'subject_id','report']].values[0].tolist()
                # report = "This is a test report."
                # report = str(info[3])
                report = clean_report_mimic_cxr(self.caption_dict[dicom_id])

                keyword = None
                if use_keywords:
                    keyword = ', '.join(self.keywords_dict[dicom_id][:key_topk]) #if use_keywords else None

                patches_vq = None
                if use_matched_patches:
                    if dicom_id in self.patches_dict:
                        topk_patch_num = min(len(self.patches_dict[dicom_id]), keypatch_topk) #if use_keypatch else keypatch_topk
                        patches_vq = [ x + tokenizer_len for x in self.patches_dict[dicom_id][:topk_patch_num]]
                
                viewPosition = None
                if use_view_position: 
                    viewPosition = meta_data[dicom_id]
                    if viewPosition in view_pos_dict:
                        viewPosition = view_pos_dict[viewPosition] + ' '
                    else: 
                        viewPosition = ''

                self.outputs.append({"report": report, 
                                    "cxr_vq_shifted": cxr_vq_shifted,
                                    "io_type": io_type,
                                    "dicom_id": dicom_id,
                                    "finding": str(info[0]),
                                    "study_id": str(info[1]),
                                    "subject_id": str(info[2]),
                                    "keyword": keyword,
                                    "patch_vq": patches_vq,
                                    "view_pos": viewPosition
                                    })
            except:
                continue
        # for i, dicom_id in enumerate(tqdm(self.dicom_ids, colour="green", unit='pair')):
        #     # if (stage == 1) or (dicom_id in self.selected_dicom_ids):
        #     if (dicom_id in self.selected_dicom_ids):
        #         try:
        #             io_type = "input" if i % 2 == 0 else "output"
        #             cxr_vq = self.cxr_vq[dicom_id]
        #             cxr_vq_shifted = [x + tokenizer_len for x in cxr_vq]
        #             # report = self._load_report(dicom_id, parse_fun_map[stage])
        #             info = df_split.loc[(df_split['dicom_id'] == dicom_id),['finding','study_id', 'subject_id', 'report']].values[0].tolist()
        #             self.outputs.append({"report": str(info[3]), 
        #                                 "cxr_vq_shifted": cxr_vq_shifted,
        #                                 "io_type": io_type,
        #                                 "dicom_id": dicom_id,
        #                                 "finding": str(info[0]),
        #                                 "study_id": str(info[1]),
        #                                 "subject_id": str(info[2]),
        #                                 })
        #         except:
        #             continue

        # total_num = len(self.selected_dicom_ids) #len(self.outputs)
        # num_each_split = int(np.ceil(total_num / split_total) )
        # start_idx = split_idx * num_each_split
        # end_idx = min((split_idx + 1 ) * num_each_split, total_num)
        # self.outputs = self.outputs[start_idx:end_idx]

        del self.db
        del self.dicom_ids
        # del self.selected_dicom_ids
        del self.cxr_vq


    def __len__(self) -> int: 
        return len(self.outputs)
        # return len(self.selected_dicom_ids)


    def __getitem__(self, idx: int): 
        return self.outputs[idx]


    def _dicom_id_to_report_path(self, dicom_id:str):
        db_series = self.db.loc[dicom_id]
        subject_id = "p" + db_series["subject_id"]
        study_id = "s" + db_series["study_id"] + ".txt"
        subject_id_prefix = subject_id[:3]

        return self.report_path / Path("files") / Path(subject_id_prefix) / Path(subject_id) / Path(study_id)
    

    def _load_report(self, dicom_id: str, parse_fun):
        report_path = self._dicom_id_to_report_path(dicom_id)
        with open(report_path, "r") as f:
            txt = f.readlines()
            
        return parse_fun(txt)
    

    def _parse_report_fi(self, txt: str) -> str:
        txt = " ".join([line.strip() for line in txt if line.strip() != ""])

        try:
            _, f_and_i = txt.split("FINDINGS:")
            try:
                f, i = f_and_i.strip().split("IMPRESSION:")
                f_and_i = f.strip() + " " + i.strip()
            except:
                f_and_i = f_and_i.strip()
        except:
            try:
                f_and_i = txt
                _, i = f_and_i.strip().split("IMPRESSION:")
                f_and_i = i.strip()
            except:
                raise ValueError

        return f_and_i
    

    def _parse_report_i(self, txt: str) -> str:
        txt = " ".join([line.strip() for line in txt if line.strip() != ""])
        
        try:
            _, impression = txt.strip().split("IMPRESSION:")
        except:
            raise ValueError
        
        return impression.strip()


    def _check_vq(self):
        error_ids = set()
        for dicom_id, cxr_vq in self.cxr_vq.items():
            if len(cxr_vq) != CXR_VQ_VQ_LEN or max(cxr_vq) >= CXR_VQ_CODE_BOOK_SIZE:
                error_ids.add(dicom_id)
        print(f"{bcolors.FAIL}[Warning] # of error dicom vq(s): {len(error_ids)}{bcolors.ENDC}")
        return error_ids
    
def sample_cxr_vq_output_instruction():
    return random.choice(CXR_VQ_OUTPUT_INSTRUCTION_LIST)

def sample_cxr_vq_input_instruction():
    return random.choice(CXR_VQ_INPUT_INSTRUCTION_LIST)

def _find_vq_replace_token_idx(input_ids: List[int], vq_replace_token_ids: List[int]):
    assert len(vq_replace_token_ids) == 3
    
    for i in range(len(input_ids)):
        if input_ids[i:i+len(vq_replace_token_ids)] == vq_replace_token_ids:
            return i
    return None

def get_inject_vq_fun(tokenizer):
    replace_tokens = tokenizer(CXR_VQ_VQ_REPLACE_TEMPLATE)['input_ids']

    def inject_vq(input_ids: List[int], cxr_vq_shifted: List[int]) -> List[int]:
        assert len(cxr_vq_shifted) == CXR_VQ_VQ_LEN
        assert max(cxr_vq_shifted) >= CXR_VQ_TOKENIZER_LEN

        first_idx = _find_vq_replace_token_idx(input_ids, replace_tokens)
        second_idx = _find_vq_replace_token_idx(input_ids[first_idx+1:], replace_tokens)

        assert first_idx is not None
        assert second_idx is None

        return input_ids[:first_idx+1] + cxr_vq_shifted + input_ids[first_idx+2:]
    
    return inject_vq

def get_extract_vq_fun(tokenizer):
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    img_token_id = tokenizer("image")['input_ids']
    response_token_id = tokenizer(RESPONSE_KEY_NL)['input_ids']
    assert len(img_token_id) == 1
    assert len(response_token_id) == 1
    img_token_id = img_token_id[0]
    response_token_id = response_token_id[0]

    def extract_vq(input_ids: List[int]):
        sequence = input_ids.clone().flatten().cpu().numpy()
        reponse_start = np.where(sequence == response_token_id)[0][0] + 1

        is_vq = sequence >= CXR_VQ_TOKENIZER_LEN
        if np.any(is_vq):
            vq_start = np.where(is_vq)[0][0]
            if vq_start >= reponse_start:
                vq = sequence[is_vq] - CXR_VQ_TOKENIZER_LEN
                if len(vq) == CXR_VQ_VQ_LEN:
                    vq = vq.tolist()
                    input_ids[..., vq_start] = img_token_id
                    return vq
                else: 
                    print(f"VQ token found but not of length {CXR_VQ_VQ_LEN}: {len(vq)}")
                    return None
            else:
                return None
        else:
            return None
    
    return extract_vq


if __name__ == "__main__":
    pass