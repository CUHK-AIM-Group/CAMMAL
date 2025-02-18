import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from mgca.constants import *
from mgca.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3, use_trainset=False):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        # DEBUG
        # self.df = self.df.iloc[0:1000]
        # self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
        #     lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # get correct image root 
        # self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
        #     lambda x: x.split("/")[1:])
        # use valid set as test set
        if split == 'test':
            split = "valid"
        print("------------use_trainset:",use_trainset)
        if use_trainset: # test on trainset 
            split = "train"
        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        print("------self.filenames:",len(self.filenames))

        # keywords list
        keywords_path = '../../materials/mimic_top500_keywords.txt'
        keyword_file = open(keywords_path, 'r')
        tmp = keyword_file.readlines()
        keyword_file.close()
        self.keyword_list = [line.strip() for line in tmp]

        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if use_trainset: # test on trainset but use test mode
            split = 'test'
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        
        self.tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(MIMIC_CXR_DATA_DIR, "captions.pickle") #os.path.join(
            # BASE_DIR, "../../data/captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in tqdm(self.df.itertuples(), total=self.df.shape[0]):
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
            # if cur_split == split and path in path2sent and os.path.exists(path):
                # print("path:",path)
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.keyword_list #self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
            return_special_tokens_mask=True,
            is_split_into_words=True
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len, sent

    def __getitem__(self, index):
        key = self.filenames[index]
        # print("key:", os.path.exists(key),key)
        # caps, cap_len, sents = self.get_caption(key)
        img_path = os.path.join(MIMIC_CXR_IMAGE_DIR, key.split('/')[-1])
        imgs = get_imgs(img_path, self.imsize, self.transform, multiscale=False)
        # return imgs, caps, cap_len, key, sents
        return imgs, self.keyword_list, key, self.tokenizer, self.max_words


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    speical_tokens_mask = []
    sentences = []
    path = []
    word_ids = []

    bs = len(batch)
    keyword_lists = [ batch[0][1] for i in range(bs)]
    tokenizer = batch[0][3]
    max_words = batch[0][4]
    tokens_out = tokenizer.batch_encode_plus(
        keyword_lists,
        return_tensors="pt",
        # truncation=True,
        padding="max_length",
        # max_length=max_words,
        return_special_tokens_mask=True,
        is_split_into_words=True,
        return_length=True
    )
    # print("tokens_out:", tokens_out)
    # print("tokens_out - inputids:",tokens_out["input_ids"])
    # print('tokens_out["length"]:',tokens_out["length"])
    word_id = tokens_out.word_ids(0)
    word_id_out = torch.tensor([-1 if idx is None else idx for idx in word_id])
    # print("word_id:",type(word_id), word_id)
    i = 0
    for b in batch:
        img, sent, p, _, _ = b
        imgs.append(img)
        cap_len.append(tokens_out["length"][i])
        ids.append(tokens_out["input_ids"][i])
        tokens.append(tokens_out["token_type_ids"][i])
        attention.append(tokens_out["attention_mask"][i])
        speical_tokens_mask.append(tokens_out["special_tokens_mask"][i])
        word_ids.append(word_id_out)
        path.append(p)
        sentences.append(sent)
        i+=1

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    speical_tokens_mask = torch.stack(speical_tokens_mask).squeeze()
    word_ids = torch.stack(word_ids).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)

    path = np.array(path)
    sentences = np.array(sentences)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices],
        "special_tokens_mask": speical_tokens_mask[sorted_cap_indices],
        "sentences": sentences[sorted_cap_indices],
        "word_ids" : word_ids[sorted_cap_indices]
    }
    return return_dict


if __name__ == "__main__":
    from mgca.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
