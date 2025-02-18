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
import os
import json
import time
import re

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

CXR_VQ_OUTPUT_INSTRUCTION_LIST_WITH_POS = [
    "Generate a {pos}chest X-ray image that corresponds to the entered free-text radiology reports for the {pos}chest X-ray image.".format(pos="{pos}"),
    "Use the free-text radiology reports for the {pos}chest X-ray image to produce a corresponding {pos}chest X-ray image.".format(pos="{pos}"),
    "Utilize the entered free-text radiology reports for the {pos}chest X-ray image to create a corresponding {pos}chest X-ray image.".format(pos="{pos}"),
    "Create a {pos}chest X-ray image that matches the free-text radiology reports entered for the {pos}chest X-ray image.".format(pos="{pos}"),
    "Produce a {pos}chest X-ray image that is consistent with the free-text radiology reports entered for the {pos}chest X-ray image.".format(pos="{pos}"),
    "Based on the free-text radiology reports for the {pos}chest X-ray image, generate a corresponding {pos}chest X-ray image.".format(pos="{pos}"),
    "Use the free-text radiology reports entered for the {pos}chest X-ray image to create a corresponding {pos}chest X-ray image.".format(pos="{pos}"),
    "Generate a {pos}chest X-ray image that is in accordance with the free-text radiology reports for the {pos}chest X-ray image entered.".format(pos="{pos}"),
    "Create a {pos}chest X-ray image that corresponds to the free-text radiology reports entered for the {pos}chest X-ray image.".format(pos="{pos}"),
    "Utilize the entered free-text radiology reports for the {pos}chest X-ray image to produce a corresponding {pos}chest X-ray image.".format(pos="{pos}")
]

CXR_VQ_VQ_REPLACE_TEMPLATE = "<image>"
CXR_VQ_VQ_KEYPATCH_REPLACE_TEMPLATE = "<keypatch>"

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
    def __init__(self, report_path: str, vq_path: str, tokenizer_len: int, mode: str, stage: int, split_path: str, \
                    only_use_findings: bool, use_origin_report: bool, use_keywords: bool, keywords_path: str, key_topk: int, \
                    use_matched_patches: bool, patches_path: str, keypatch_topk: int, use_flag: bool, use_view_position: bool):
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
        # load captions
        self.caption_dict = None
        if use_origin_report:
            caption_path = os.path.join(split_path, "mimic_xray_"+mode+"_caption.pickle")
            with open(caption_path, "rb") as f:
                self.caption_dict = cPickle.load(f)
        
        # load meta data
        meta_data = None
        if use_view_position:
            meta_data = pd.read_csv(self.split_path / "mimic-cxr-2.0.0-metadata.csv", index_col='dicom_id', dtype=str)
            meta_data = meta_data[['ViewPosition']]
            meta_data = meta_data.to_dict()['ViewPosition']
            view_pos_dict = {'LATERAL': 'lateral', 'PA': 'PA' , 'AP': 'AP', 'LL': 'left lateral'}


        # self.dicom_ids = self.dicom_ids[:1000]

        # with open(self.split_path / "mimic-cxr-2.0.0-selected-pa-ap-earlist-study.pickle", "rb") as f:
        #     self.selected_dicom_ids = set(cPickle.load(f).keys())

        split_file = "mimic-cxr-2.0.0-"+ mode +".csv"
        df_split = pd.read_csv(self.split_path / split_file)
        self.selected_dicom_ids = df_split['dicom_id'].tolist()

        # self.selected_dicom_ids = self.selected_dicom_ids[:1000]
        # print("-------self.selected_dicom_ids:",len(self.selected_dicom_ids))

        self.findings_list = None
        if only_use_findings:
            finding_root = 'data_root/dataset/x_ray/mimic_xray/anno_clean'
            file_list = os.listdir(finding_root)
            self.findings_list = {}
            for filename in file_list:
                file_path = os.path.join(finding_root, filename)
                data_list = json.loads(open(file_path, 'r').read())
                for item in data_list:
                    dicom_id = item['image_id']
                    self.findings_list[dicom_id] = item['caption']
            print("========== Only use findings as report ============")
        # print("=========================only_use_findings:", only_use_findings,"use_origin_report:",use_origin_report,"use_view_position:",use_view_position)

        self.outputs = [] 
        # keypatch_topk = 5
        for i, dicom_id in enumerate(tqdm(self.selected_dicom_ids, colour="green", unit='pair')):
            # if (stage == 1) or (dicom_id in self.selected_dicom_ids):
            # if (dicom_id in self.selected_dicom_ids):
            # try:
            start_time = time.time()
            # io_type = "input" if i % 2 == 0 else "output"
            cxr_vq = self.cxr_vq[dicom_id]
            cxr_vq_shifted = [x + tokenizer_len for x in cxr_vq]
            # time_0 = time.time()
            # print("time - cxr_vq:", time_0-start_time)
            if only_use_findings:
                report = self.findings_list[dicom_id]
            elif use_origin_report:
                report = self.caption_dict[dicom_id]#df_split.loc[(df_split['dicom_id'] == dicom_id),['report']].values[0][0]
            else:
                report = self._load_report(dicom_id, parse_fun_map[stage])
            # time_1 = time.time()
            # print("time - report:", time_1-time_0)
            report = clean_report_mimic_cxr(report)

            patches_vq = None
            # if use_matched_patches and io_type == "output":
            if use_matched_patches:
                if dicom_id in self.patches_dict:
                    topk_patch_num = min(len(self.patches_dict[dicom_id]), keypatch_topk) #if use_keypatch else keypatch_topk
                    patches_vq = [ x + tokenizer_len for x in self.patches_dict[dicom_id][:topk_patch_num]]
            # time_2 = time.time()
            # print("time - keypatch:", time_2-time_1)

            keyword = None
            # if use_keywords and io_type == "input" and dicom_id in self.keywords_dict:
            if use_keywords and dicom_id in self.keywords_dict:
                topk_num = min(len(self.keywords_dict[dicom_id]), key_topk) #if use_keywords else key_topk
                keyword = ', '.join(self.keywords_dict[dicom_id][:topk_num]) #if use_keywords else None
            # time_3 = time.time()
            # print("time - keywords:", time_3-time_2)

            viewPosition = None
            # if use_view_position and io_type == "output":
            if use_view_position:
                viewPosition = meta_data[dicom_id]
                if viewPosition in view_pos_dict:
                    viewPosition = view_pos_dict[viewPosition] + ' '
                else: 
                    viewPosition = ''
            # time_4 = time.time()
            # print("time - viewPos:", time_4-time_3)
            # time_2 = time.time()
            # print("------keyword:", keyword)

            io_type = "input" 
            self.outputs.append({"report": report, 
                                "cxr_vq_shifted": cxr_vq_shifted,
                                "io_type": io_type,
                                "keyword": keyword,
                                "patch_vq": patches_vq,
                                "view_pos": viewPosition})
            io_type = "output"
            self.outputs.append({"report": report, 
                                "cxr_vq_shifted": cxr_vq_shifted,
                                "io_type": io_type,
                                "keyword": keyword,
                                "patch_vq": patches_vq,
                                "view_pos": viewPosition})
            # if i < 10:
            #     print("time - cxr_vq:", time_0-start_time, "report:", time_1-time_0,"keypatch:", time_2-time_1,"keywords:", time_3-time_2,"viewPos:", time_4-time_3)
                # print("total:",time_2-start_time,"vq:", time_0-start_time, "report:", time_1-time_0, "keyword:",time_2-time_1)
            # except:
            #     continue
        ## DEBUG
        if mode == "train":
            print("trainset:",len(self.outputs))
        #     with open('./tmp/mimic_cxr_trainset.pkl', 'wb') as file:
        #         # A new file will be created
        #         cPickle.dump(self.outputs, file)
        
        del self.db
        del self.dicom_ids
        # del self.selected_dicom_ids
        del self.cxr_vq
        del self.keywords_dict

    def __len__(self) -> int: 
        return len(self.outputs)


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

def sample_cxr_vq_output_instruction_with_position(position):
    out = random.choice(CXR_VQ_OUTPUT_INSTRUCTION_LIST_WITH_POS)
    return out.format(pos=position)

def sample_cxr_vq_input_instruction():
    return random.choice(CXR_VQ_INPUT_INSTRUCTION_LIST)

def _find_vq_replace_token_idx(input_ids: List[int], vq_replace_token_ids: List[int], len_token_ids: int):
    # assert len(vq_replace_token_ids) == 3
    assert len(vq_replace_token_ids) == len_token_ids
    
    for i in range(len(input_ids)):
        if input_ids[i:i+len(vq_replace_token_ids)] == vq_replace_token_ids:
            return i
    return None

def get_inject_vq_fun(tokenizer, use_keypatch=False, topk_patch_num=5):
    replace_tokens = tokenizer(CXR_VQ_VQ_REPLACE_TEMPLATE)['input_ids']
    replace_tokens_keypatch = tokenizer(CXR_VQ_VQ_KEYPATCH_REPLACE_TEMPLATE)['input_ids']
    pad_token = tokenizer(tokenizer.pad_token)['input_ids']
    topk_patch = topk_patch_num

    def inject_vq(input_ids: List[int], cxr_vq_shifted: List[int]) -> List[int]:
        assert len(cxr_vq_shifted) == CXR_VQ_VQ_LEN
        assert max(cxr_vq_shifted) >= CXR_VQ_TOKENIZER_LEN

        first_idx = _find_vq_replace_token_idx(input_ids, replace_tokens, 3)
        second_idx = _find_vq_replace_token_idx(input_ids[first_idx+1:], replace_tokens, 3)

        assert first_idx is not None
        assert second_idx is None

        return input_ids[:first_idx+1] + cxr_vq_shifted + input_ids[first_idx+2:]
    
    def inject_vq_keypatch(input_ids: List[int], cxr_vq_shifted: List[int], patch_vq: List[int]) -> List[int]:
        assert len(cxr_vq_shifted) == CXR_VQ_VQ_LEN
        assert max(cxr_vq_shifted) >= CXR_VQ_TOKENIZER_LEN

        first_idx = _find_vq_replace_token_idx(input_ids, replace_tokens, 3)
        second_idx = _find_vq_replace_token_idx(input_ids[first_idx+1:], replace_tokens, 3)

        assert first_idx is not None
        assert second_idx is None

        input_ids = input_ids[:first_idx+1] + cxr_vq_shifted + input_ids[first_idx+2:]

        # keypatch
        first_idx = _find_vq_replace_token_idx(input_ids, replace_tokens_keypatch, 4)
        if first_idx is None:
            out = input_ids #+ pad_token*topk_patch
            # print("no keypatch:", len(out))
            return out, first_idx
        second_idx = _find_vq_replace_token_idx(input_ids[first_idx+1:], replace_tokens_keypatch, 4)

        # print("---------replace_tokens_keypatch:",len(replace_tokens_keypatch),replace_tokens_keypatch)
        assert first_idx is not None
        assert second_idx is None

        out = input_ids[:first_idx+1] + patch_vq + input_ids[first_idx+3:]
        # print("with keypatch:", len(out))
        return out, first_idx
    
    if use_keypatch:
        return inject_vq_keypatch
    return inject_vq

def get_inject_vq_fun_keypatch(tokenizer):
    replace_tokens_keypatch = tokenizer(CXR_VQ_VQ_KEYPATCH_REPLACE_TEMPLATE)['input_ids']

    def inject_vq_keypatch(input_ids: List[int], patch_vq: List[int]) -> List[int]:
        # keypatch
        first_idx = _find_vq_replace_token_idx(input_ids, replace_tokens_keypatch, 4)
        second_idx = _find_vq_replace_token_idx(input_ids[first_idx+1:], replace_tokens_keypatch, 4)

        assert first_idx is not None
        assert second_idx is None

        return input_ids[:first_idx+1] + patch_vq + input_ids[first_idx+3:]
    
    return inject_vq_keypatch

def get_extract_vq_fun_new(tokenizer, use_keypatch=False):
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    img_token_id = tokenizer("image")['input_ids']
    response_token_id = tokenizer(RESPONSE_KEY_NL)['input_ids']
    assert len(img_token_id) == 1
    assert len(response_token_id) == 1
    img_token_id = img_token_id[0]
    response_token_id = response_token_id[0]
    # print("response_token_id:",response_token_id)
    def extract_vq(input_ids: List[int]):
        sequence = input_ids.clone().flatten().cpu().numpy()
        # print("sequence:",sequence)
        reponse_start = np.where(sequence == response_token_id)[0][0] + 1
        # print("reponse_start:",reponse_start)
        is_vq = sequence >= CXR_VQ_TOKENIZER_LEN
        if np.any(is_vq):
            vq_start = np.where(is_vq)[0][0]
            # print("vq_start:",vq_start)
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
                if use_keypatch:
                    sequence = sequence[reponse_start:]
                    # print(sequence)
                    is_vq = sequence >= CXR_VQ_TOKENIZER_LEN
                    # print("is_vq:",is_vq)
                    if np.any(is_vq):
                        vq = sequence[is_vq] - CXR_VQ_TOKENIZER_LEN
                        # print("len(vq):",len(vq),"CXR_VQ_VQ_LEN:",CXR_VQ_VQ_LEN)
                        if len(vq) == CXR_VQ_VQ_LEN:
                            vq = vq.tolist()
                            input_ids[..., vq_start] = img_token_id
                            return vq
                        elif len(vq) > CXR_VQ_VQ_LEN:
                            vq = vq[:CXR_VQ_VQ_LEN]
                            vq = vq.tolist()
                            input_ids[..., vq_start] = img_token_id
                            return vq
                        else: 
                            print(f"VQ token found but not of length {CXR_VQ_VQ_LEN}: {len(vq)}")
                            return None
                        # print("vq_start >= reponse_start:",vq_start >= reponse_start)
                    else:
                        return None
                return None
        else:
            return None
    
    return extract_vq

def get_extract_vq_fun(tokenizer):
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    img_token_id = tokenizer("image")['input_ids']
    response_token_id = tokenizer(RESPONSE_KEY_NL)['input_ids']
    assert len(img_token_id) == 1
    assert len(response_token_id) == 1
    img_token_id = img_token_id[0]
    response_token_id = response_token_id[0]
    print("response_token_id:",response_token_id)
    def extract_vq(input_ids: List[int]):
        sequence = input_ids.clone().flatten().cpu().numpy()
        reponse_start = np.where(sequence == response_token_id)[0][0] + 1
        print("reponse_start:",reponse_start)
        is_vq = sequence >= CXR_VQ_TOKENIZER_LEN
        if np.any(is_vq):
            vq_start = np.where(is_vq)[0][0]
            print("vq_start:",vq_start)
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
                print("vq_start >= reponse_start:",vq_start >= reponse_start)
                return None
        else:
            return None
    
    return extract_vq

if __name__ == "__main__":
    pass