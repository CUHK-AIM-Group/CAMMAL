from training.generate import generate_response, load_model_tokenizer_for_generate, load_model_tokenizer_for_generate_separate
from training.mimiccxr_vq_dataset_new import sample_cxr_vq_output_instruction, sample_cxr_vq_input_instruction, CXR_VQ_TOKENIZER_LEN
from training.mimiccxr_vq_dataset import sample_cxr_vq_output_instruction_with_position

from training.mimiccxr_vq_dataset_new import (
    MimicCxrVqDataset, 
    get_inject_vq_fun,
    CXR_VQ_VQ_REPLACE_TEMPLATE,
    CXR_VQ_CODE_BOOK_SIZE, 
    CXR_VQ_VQ_LEN
)

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    set_seed,
)
from typing import List
from argparse import ArgumentParser

import logging
from torch.utils.data.dataloader import DataLoader
import datasets

from datasets import Dataset, load_dataset

from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from training.consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_WITH_INPUT_KEYWORD_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
)
from tqdm import tqdm

from taming.models.vqgan import GumbelVQ, VQModel
import numpy as np
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from utils import metrics
import time
from training.openI_dataset_new import OpenIDataset
logger = logging.getLogger(__name__)


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def shift_vq_tokens(tokens: List[int], tokenizer: PreTrainedTokenizer) -> List[int]:
    assert len(tokenizer) == CXR_VQ_TOKENIZER_LEN
    return [token + len(tokenizer) for token in tokens]


def load_test_dataset_mimiccxrvq(tokenizer_len, stage, split_idx, split_total, dataset_name, use_keywords, keywords_path, key_topk, use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position) -> Dataset:
    if dataset_name == 'mimic-cxr':
        logger.info(f"Loading dataset from MIMIC-CXR-VQ-Dataset")
        # dataset_train = MimicCxrVqDataset("data/mimic-cxr-reports", 
        #                                   "data/mimiccxr_vqgan1024_res256_3e_codebook_indices.pickle", 
        #                                   tokenizer_len, 
        #                                   "train",
        #                                   stage)
        dataset_test  = MimicCxrVqDataset("~/dataset/x_ray/mimic_xray/mimic-cxr-reports", 
                                        "~/dataset/x_ray/mimic_xray/info/mimiccxr_vqgan1024_res256_3e_codebook_indices.pickle", 
                                        tokenizer_len, 
                                        "test",
                                        stage,
                                        "~/dataset/x_ray/mimic_xray/info",
                                        split_idx, split_total, use_keywords, keywords_path, key_topk, use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)
    elif dataset_name == 'openI':
        logger.info(f"Loading dataset from OpenI-Dataset")
        codebook_path = "~/dataset/x_ray/openI/anno/openI_vqgan1024_codebook_all.pickle"
        # codebook_path = "~/dataset/x_ray/openI/data/openI_vqgan1024_codebook_all_mimic_model.pickle"
        dataset_test  = OpenIDataset("", 
                                        codebook_path, 
                                        tokenizer_len, 
                                        "test",
                                        stage,
                                        "~/dataset/x_ray/openI/anno/",
                                        split_idx, split_total, use_keywords, keywords_path, key_topk, use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)
                                        # split_idx=split_idx, split_total=split_total, use_split=True)
    else:
        print("Please give a correct dataset name: mimic-cxr or openI.")
        exit(0)
    # dataset_train = Dataset.from_list(dataset_train)
    dataset_test = Dataset.from_list(dataset_test)

    logger.info("Found %d rows", dataset_test.num_rows)

    def _add_text(rec):
        if rec['io_type'] == 'input':
            if rec["keyword"] is not None:
                rec["text"] = PROMPT_WITH_INPUT_KEYWORD_FORMAT.format(instruction=sample_cxr_vq_input_instruction(), 
                                                        response=rec['report'], 
                                                        input=CXR_VQ_VQ_REPLACE_TEMPLATE,
                                                        keyword=rec['keyword'])
            else:
                rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=sample_cxr_vq_input_instruction(), 
                                                        response=rec['report'], 
                                                        input=CXR_VQ_VQ_REPLACE_TEMPLATE)
        elif rec['io_type'] == 'output':
            output_instruction = sample_cxr_vq_output_instruction_with_position(rec["view_pos"]) if ("view_pos" in rec and rec["view_pos"] is not None) else sample_cxr_vq_output_instruction()
            if "patch_vq" in rec and rec["patch_vq"] is not None:
                rec["text"] = PROMPT_WITH_INPUT_KEYPATCH_FORMAT.format(instruction=output_instruction, 
                                                        response=CXR_VQ_VQ_REPLACE_TEMPLATE, 
                                                        input=rec['report'],
                                                        keypatch=CXR_VQ_VQ_KEYPATCH_REPLACE_TEMPLATE)
            else:
                rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=output_instruction, 
                                                        response=CXR_VQ_VQ_REPLACE_TEMPLATE, 
                                                        input=rec['report'])
        else:
            raise ValueError(f"Unexpected io_type: {rec['io_type']}")

        return rec

    # dataset_train = dataset_train.map(_add_text, remove_columns="report")
    # dataset_test = dataset_test.map(_add_text, remove_columns="report")

    return dataset_test


def preprocess_batch_mimiccxrvq(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int, inject_vq_fun) -> dict:
    tokenizer_output = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

    input_idss = [inject_vq_fun(ii, vq) for ii, vq in zip(tokenizer_output["input_ids"], batch['cxr_vq_shifted'])]
    attention_masks = [[1 for _ in range(len(am) + CXR_VQ_VQ_LEN - 1)] for am in tokenizer_output["attention_mask"]]

    batch["input_ids"] = input_idss
    batch["attention_mask"] = attention_masks

    return batch


def preprocess_dataset_mimiccxrvq(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED, stage=None, split_idx=0, split_total=10, dataset_name='mimic-cxr', \
                                    use_keywords=False, keywords_path="", key_topk=10, use_matched_patches=False, patches_path="", keypatch_topk=5, use_it=False, \
                                    use_view_position=False) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset_test = load_test_dataset_mimiccxrvq(len(tokenizer), stage, split_idx, split_total, dataset_name, use_keywords, keywords_path, key_topk, \
                                                                            use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)

    '''logger.info("Preprocessing dataset MIMIC-CXR-VQ")
    _preprocessing_function = partial(preprocess_batch_mimiccxrvq, 
                                      max_length=max_length, 
                                      tokenizer=tokenizer, 
                                      inject_vq_fun=get_inject_vq_fun(tokenizer))
    # dataset_train = dataset_train.map(
    #     _preprocessing_function,
    #     batched=True,
    #     remove_columns=["cxr_vq_shifted", "text", "io_type"],
    # )
    dataset_test = dataset_test.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["cxr_vq_shifted", "text", "io_type"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset MIMIC-CXR-VQ has %d rows",  dataset_test.num_rows)
    # dataset_train = dataset_train.filter(lambda rec: len(rec["input_ids"]) < max_length)
    dataset_test = dataset_test.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset MIMIC-CXR-VQ has %d rows after filtering for truncated records", dataset_test.num_rows)
    logger.info("Done preprocessing MIMIC-CXR_VQ")'''

    return  dataset_test


def main():
    parser = ArgumentParser()
    # parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--split_idx", type=int, required=True)
    parser.add_argument("--split_total", type=int, required=False, default=6)
    parser.add_argument("--dataset_name", type=str, default='mimic-cxr')

    parser.add_argument("--use-keywords", type=bool, default=False, help="Whether to use keywords.")
    parser.add_argument("--keywords-path", type=str, default="~/code/img2text2img/MGCA/data/topk_word_list/pickles/2023_07_29_21_39_42_keywords_top10.pickle", help="path for keywords pickle.")
    parser.add_argument("--key-topk", type=int, default=10, help="topk keywords to use.")

# matched patches
    parser.add_argument("--use-matched-patches", type=bool, default=False, help="Whether to use matched patches.")
    parser.add_argument("--patches-path", type=str, default="~/code/img2text2img/MGCA/data/topk_patch_list/pickles/2023_08_14_07_26_14_patches_matched_top20_patches_for_all_reports_dict.pickle", help="path for matched patches.")

    parser.add_argument("--keypatch-topk", type=int, default=5, help="topk keypatches to use.")
    parser.add_argument("--use-it", type=bool, default=False, help="Whether to use it.")
    parser.add_argument("--use-view-position", type=bool, default=False, help="Whether to use view position for cxr generation.")

    args = parser.parse_args()

    save_root = args.results_path #os.path.join('./results', '')
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_img_root = os.path.join(args.results_path, 'images')
    if not os.path.exists(save_img_root):
        os.mkdir(save_img_root)
    
    split_idx = args.split_idx
    split_total = args.split_total
    results_csv_root = os.path.join(save_root, 'outputs_'+str(split_idx)+'_'+ str(split_total)+'.csv')
    # logger

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", filename=os.path.join(save_root, 'log_'+str(split_idx)+'_'+ str(split_total)+'.txt')
    )
    
    model, tokenizer = load_model_tokenizer_for_generate(args.model_path)
    
    model = model.cuda()
    # tokenizer = tokenizer.cuda()
    # define VQ-GAN model

    PATH_CONFIG = "./pretrained_model/mimiccxr_vqgan1024_res256_3e_ckpts/2023-05-11T23-37-27-project.yaml" #"<path to the trained model config (.yaml)>"
    PATH_CKPT = "./pretrained_model/mimiccxr_vqgan1024_res256_3e_ckpts/last-3e.ckpt" #"<path to the trained model ckpts (.ckpt)>"
    # if args.dataset_name == 'openI':
    #     root = '~/code/image2text2image/CXR2Report2CXR/pretrained_model/2023-06-27T20-20-29_custom_vqgan/'
    #     PATH_CONFIG = os.path.join(root, 'configs/2023-06-27T20-20-29-project.yaml')
    #     PATH_CKPT = os.path.join(root, 'checkpoints/last.ckpt')
    
    torch.set_grad_enabled(False)

    config = load_config(PATH_CONFIG, display=False)
    vqgan_model = load_vqgan(config, ckpt_path=PATH_CKPT, is_gumbel=False).cuda() #to(DEVICE)


    # load dataset
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")

    stage = 1 
    batch_size = 1 #4
    seed = DEFAULT_SEED
    set_seed(seed)
    split_dataset_mimiccxrvq_test = preprocess_dataset_mimiccxrvq(tokenizer=tokenizer, max_length=max_length, seed=seed, stage=stage, split_idx=split_idx, split_total=split_total, dataset_name=args.dataset_name, \
                                                                    use_keywords=args.use_keywords, keywords_path=args.keywords_path, key_topk=args.key_topk, \
                                                                        use_matched_patches=args.use_matched_patches, patches_path=args.patches_path, keypatch_topk=args.keypatch_topk, use_it=args.use_it, \
                                                                            use_view_position=args.use_view_position)
   
    test_loader = DataLoader(split_dataset_mimiccxrvq_test, batch_size=batch_size,shuffle=False, num_workers=0,drop_last=False, pin_memory=True)
    generated_list = []

    gt_dict = {}
    pred_dict = {}
    error_list = []
    j = 0
    for i, batch in enumerate(tqdm(test_loader, total=len(test_loader))): 
        dicom_id = batch["dicom_id"][0]
        try:
            # print("batch:",type(batch))
            # print(batch)
            # break
            start_time_0 = time.time()
            instruction_text_cxr2report = sample_cxr_vq_input_instruction()
            view_pos = batch['view_pos'][0]
            if view_pos and view_pos != 'none':
                instruction_text_report2cxr = sample_cxr_vq_output_instruction_with_position(view_pos)
            else:
                instruction_text_report2cxr = sample_cxr_vq_output_instruction()

            # print("-----------------------------------", dicom_id ,"---------------------------------------------")
            # CXR-to-Report
            # print("Start to CXR-to-report....")
            input_vq = batch['cxr_vq_shifted']
            input_key = batch['keyword'][0]
            generated_text, _ = generate_response((instruction_text_cxr2report, input_vq, input_key), model=model, tokenizer=tokenizer, max_new_tokens=512, use_keypatch=args.use_matched_patches)
            # if response:
            #     print(f"----------- {i}/{len(test_loader)}\n\nInstruction: {instruction_text_cxr2report}\n\nInput: {input_text}\n\nResponse: {response}\n\nGenerated-VQ: {response_vq}\n\n-----------\n\n\n")
            # print("generated_text:",generated_text)
            # Report-to-CXR 
            # print("Start to report-to-CXR......")
            input_text = batch['report'][0]
            input_keypatch = batch['patch_vq']
            _, generated_vq = generate_response((instruction_text_report2cxr, input_text, input_keypatch), model=model, tokenizer=tokenizer, max_new_tokens=512, use_keypatch=args.use_matched_patches)
            # print("generated_vq:", len(generated_vq), generated_vq)
            # if response:
            #     print(f"----------- {i}/{len(test_loader)}\n\nInstruction: {instruction_text_cxr2report}\n\nInput: {input_text}\n\nResponse: {response}\n\nGenerated-VQ: {response_vq}\n\n-----------\n\n\n")

            if args.dataset_name == 'mimic-cxr':
                study_id = batch["study_id"][0]
                subject_id = batch["subject_id"][0]
                dicom_id = batch["dicom_id"][0]
                finding = batch["finding"][0]
                img_path = os.path.join(save_img_root, subject_id + "_" + study_id + "_" + dicom_id + '_'+ view_pos + '.jpg')
                tmp = {
                    "dicom_id" : dicom_id,
                    "finding" : finding,
                    "report" : generated_text,
                    "img_path" : img_path,
                    "real_report": input_text,
                    "keyword" : input_key
                }
            elif args.dataset_name == 'openI':
                # study_id = batch["study_id"][0]
                # subject_id = batch["subject_id"][0]
                dicom_id = batch["dicom_id"][0]
                # finding = batch["finding"][0]
                img_path = os.path.join(save_img_root, 'CXR'+ dicom_id + '_'+ view_pos +'.jpg') #subject_id + "_" + study_id + "_" + dicom_id + '.jpg')
                tmp = {
                    "dicom_id" : dicom_id,
                    # "finding" : finding,
                    "report" : generated_text,
                    "img_path" : img_path,
                    "real_report": input_text,
                    "keyword" : input_key
                }
            else: 
                print("Please give the correct dataset name, e.g. mimic-cxr or openI")
            generated_list.append(tmp)
            # save image
            start_time = time.time()
            indices = torch.tensor(generated_vq).cuda()#.to(DEVICE)
            img = vqgan_model.decode(vqgan_model.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1)))
            img = img.squeeze().permute(1,2,0).cpu().numpy()
            img = np.clip(img, -1., 1.)
            img = (img + 1.)/2.
            plt.imsave(img_path, img)
            end_time = time.time()
            # print("duration -", "vqgan:", end_time-start_time,"total:", end_time-start_time_0)


            # gt_dict[i] = [input_text] 
            # pred_dict[i] = [generated_text] 
            gt_dict[j] = [input_text] 
            pred_dict[j] = [generated_text] 
            j += 1
        except:
            error_list.append(dicom_id)
            continue
    print("len(error_list):", len(error_list))
    print("error_list:",error_list)
    gen_df = pd.DataFrame(generated_list)
    gen_df.to_csv(results_csv_root)
    
    test_met = metrics.compute_scores(gt_dict, pred_dict)
    for k, v in test_met.items():
        logger.info("{}\t{}".format(k, v)) 


    # iter_wrapper = (lambda x: tqdm(x, total=len(test_loader)))

    # for i, data_iter in iter_wrapper(enumerate(test_loader)):
    #     print("data_iter:",type(data_iter))
    #     print("keys:",data_iter.keys())
    #     # print("output:",data_iter)
    #     report = data_iter["report"]
    #     cxr_vq_shifted = data_iter["cxr_vq_shifted"]
    #     print("report:",report)
    #     print("cxr_vq_shifted:",cxr_vq_shifted.shape)

'''

    instruction_texts = [
        # Nautral language instructions
        "Tell me about Independence Day in the United States.",
        "Does MRI pose a risk of radiation exposure? Please write as long as you can.",

        # CXR-to-Report instructions
        sample_cxr_vq_input_instruction(),
        sample_cxr_vq_input_instruction(),
        sample_cxr_vq_input_instruction(),
        sample_cxr_vq_input_instruction(),

        # Report-to-CXR instructions
        sample_cxr_vq_output_instruction(),
        sample_cxr_vq_output_instruction(),
        sample_cxr_vq_output_instruction()
    ]
    input_texts = [
        None,
        None,

        # dicom_id: edf1e5ad-e7249deb-2d881608-aa2878c8-e22288bd
        [955, 245, 63, 127, 981, 1002, 829, 147, 665, 716, 447, 973, 533, 329, 659, 151, 61, 127, 410, 920, 439, 203, 600, 921, 202, 573, 742, 885, 687, 71, 22, 807, 22, 621, 764, 764, 66, 742, 742, 716, 807, 551, 860, 410, 15, 144, 719, 1012, 401, 764, 87, 122, 339, 716, 258, 144, 193, 551, 203, 122, 333, 764, 428, 921, 634, 921, 860, 245, 961, 637, 973, 345, 665, 63, 909, 1012, 200, 905, 322, 680, 133, 660, 322, 339, 551, 699, 22, 621, 87, 742, 250, 961, 127, 845, 333, 1012, 77, 295, 205, 203, 82, 71, 144, 468, 202, 807, 1012, 333, 69, 331, 144, 680, 468, 921, 331, 981, 955, 232, 1002, 467, 446, 551, 410, 716, 533, 845, 406, 147, 260, 514, 331, 533, 243, 955, 468, 634, 534, 742, 127, 1012, 921, 119, 600, 981, 22, 534, 528, 127, 807, 973, 634, 401, 77, 345, 404, 932, 528, 814, 243, 71, 250, 808, 885, 716, 932, 447, 634, 790, 999, 345, 428, 243, 406, 814, 514, 637, 659, 845, 1012, 468, 790, 421, 790, 468, 243, 34, 719, 824, 193, 163, 842, 250, 77, 193, 930, 295, 105, 119, 790, 119, 439, 428, 829, 147, 660, 401, 529, 699, 200, 808, 30, 133, 30, 529, 151, 200, 845, 202, 845, 660, 66, 331, 82, 961, 818, 467, 61, 785, 108, 955, 818, 163, 687, 329, 533, 528, 330, 15, 329, 961, 71, 1012, 764, 439, 304, 1012, 860, 596, 163, 842, 999, 845, 905, 514, 716, 122], 
        # dicom_id: 494f62af-2213616c-20174f23-c3d781fd-fed10e18
        [66, 260, 379, 555, 304, 63, 222, 127, 716, 258, 932, 1012, 34, 533, 885, 1012, 163, 163, 119, 63, 82, 860, 447, 383, 260, 383, 404, 1015, 15, 860, 260, 87, 981, 447, 339, 322, 105, 1015, 764, 764, 15, 905, 807, 232, 1015, 528, 1012, 127, 973, 764, 981, 600, 87, 15, 202, 829, 905, 845, 829, 932, 551, 410, 447, 999, 447, 203, 1012, 1002, 829, 1015, 829, 660, 860, 119, 764, 447, 468, 860, 322, 71, 1015, 370, 955, 621, 1002, 34, 202, 978, 22, 790, 119, 428, 534, 447, 401, 127, 63, 930, 932, 34, 829, 1002, 193, 660, 824, 529, 359, 534, 716, 829, 304, 63, 379, 551, 932, 764, 260, 129, 428, 921, 163, 200, 921, 785, 404, 63, 706, 555, 551, 961, 932, 764, 973, 406, 1012, 555, 200, 61, 77, 845, 860, 107, 295, 1012, 379, 203, 243, 764, 773, 687, 909, 446, 66, 163, 773, 406, 105, 333, 69, 716, 294, 66, 243, 764, 814, 331, 295, 304, 30, 193, 533, 785, 814, 292, 69, 1012, 151, 573, 719, 829, 528, 785, 814, 107, 292, 292, 193, 467, 193, 687, 83, 1012, 250, 999, 193, 22, 295, 905, 331, 203, 77, 77, 83, 193, 330, 222, 860, 63, 961, 961, 379, 600, 845, 345, 108, 200, 133, 66, 30, 108, 66, 660, 621, 339, 250, 699, 258, 534, 660, 845, 534, 551, 232, 77, 401, 359, 119, 34, 15, 329, 329, 742, 773, 845, 842, 634, 193, 421, 133, 22, 829, 379, 119, 331, 34, 339], 
        # dicom_id: c710e145-280390c3-5b9ddcf7-faa611b8-b39e60c8
        [250, 122, 66, 921, 680, 514, 295, 428, 921, 63, 981, 203, 122, 719, 410, 551, 108, 699, 955, 406, 514, 202, 1015, 406, 66, 63, 555, 909, 383, 243, 824, 808, 596, 410, 533, 133, 764, 232, 501, 807, 716, 447, 222, 202, 932, 329, 294, 573, 961, 200, 359, 955, 250, 555, 716, 439, 961, 203, 122, 447, 873, 873, 322, 555, 978, 665, 127, 773, 529, 873, 1012, 339, 742, 250, 250, 127, 829, 144, 333, 533, 555, 83, 699, 981, 232, 410, 34, 329, 250, 961, 329, 250, 659, 660, 680, 129, 842, 932, 203, 63, 410, 71, 665, 83, 699, 955, 961, 551, 66, 660, 665, 921, 842, 15, 773, 529, 909, 133, 295, 428, 961, 955, 981, 250, 790, 144, 514, 921, 428, 719, 243, 77, 807, 467, 842, 842, 304, 71, 250, 930, 82, 659, 842, 842, 660, 15, 600, 22, 322, 34, 845, 345, 428, 232, 909, 818, 63, 447, 814, 467, 905, 600, 467, 69, 845, 534, 785, 634, 764, 34, 105, 706, 1002, 921, 814, 808, 331, 468, 687, 660, 932, 163, 905, 790, 790, 528, 687, 534, 829, 105, 200, 808, 842, 596, 534, 528, 406, 330, 785, 790, 330, 468, 905, 785, 814, 808, 163, 66, 829, 133, 383, 292, 322, 621, 200, 133, 66, 133, 331, 790, 61, 719, 250, 930, 401, 193, 446, 621, 108, 108, 30, 61, 61, 61, 250, 205, 77, 77, 193, 22, 905, 955, 30, 330, 30, 292, 292, 292, 133, 330, 955, 258, 824, 955, 330, 634], 
        # dicom_id: 1bc3d3de-cd13c1cd-ce13e61d-5191632c-e3ae7b5c
        [719, 421, 551, 421, 742, 122, 680, 978, 921, 129, 329, 339, 200, 63, 814, 379, 151, 885, 706, 529, 428, 905, 144, 829, 147, 406, 447, 905, 533, 151, 978, 773, 200, 147, 978, 773, 232, 1012, 1012, 829, 534, 105, 716, 621, 15, 222, 406, 359, 439, 773, 127, 829, 637, 203, 379, 534, 331, 905, 742, 981, 764, 508, 203, 243, 151, 15, 508, 873, 71, 250, 370, 331, 193, 764, 501, 410, 383, 119, 596, 508, 22, 807, 147, 909, 410, 742, 468, 885, 706, 596, 329, 260, 659, 814, 468, 885, 151, 921, 322, 637, 329, 551, 119, 359, 596, 514, 339, 699, 955, 404, 127, 370, 151, 147, 1015, 63, 329, 528, 905, 404, 824, 34, 119, 122, 955, 1002, 807, 999, 955, 428, 1012, 551, 243, 932, 468, 446, 421, 107, 428, 528, 621, 660, 406, 63, 151, 69, 706, 401, 860, 932, 383, 151, 330, 34, 467, 69, 467, 905, 1012, 961, 294, 534, 428, 468, 706, 829, 66, 292, 329, 808, 193, 764, 814, 659, 82, 245, 250, 428, 808, 742, 790, 133, 108, 30, 383, 404, 439, 331, 1015, 222, 447, 501, 250, 660, 15, 205, 30, 446, 930, 331, 22, 87, 634, 764, 634, 222, 447, 370, 955, 1002, 660, 1002, 790, 232, 15, 63, 764, 1002, 467, 401, 660, 428, 339, 501, 529, 807, 428, 845, 955, 421, 706, 383, 528, 860, 200, 30, 842, 773, 203, 501, 699, 203, 665, 447, 69, 660, 621, 706, 232, 205, 845, 660, 921, 764, 329, 379],
        
        "A new dual-lead pacemaker with lead positioned through the left transvenous approach end into the right ventricle and is appropriate. No focal lung opacities concerning for pneumonia.  Heart is top normal size. Mediastinal and hilar contours are normal.  No evidence of pneumothorax."
        "Bilateral, diffuse, confluent pulmonary opacities. Differential diagnosis include  severe pulmonary edema or ARDS or hemorrhage. Concurrent lung infection cannot be ruled out.",
        "No acute cardiopulmonary process."
    ]

    generation_samples = list(zip(instruction_texts, input_texts))
    for i, (instruction_text, input_text) in enumerate(generation_samples, start=1):
        # Input image token must be shifted by tokenizer length before being passed to model.
        if isinstance(input_text, list):
            input_text = shift_vq_tokens(input_text, tokenizer)

        response, response_vq = generate_response((instruction_text, input_text), model=model, tokenizer=tokenizer, max_new_tokens=512)
        if response:
            print(f"----------- {i}/{len(generation_samples)}\n\nInstruction: {instruction_text}\n\nInput: {input_text}\n\nResponse: {response}\n\nGenerated-VQ: {response_vq}\n\n-----------\n\n\n")
'''

if __name__ == "__main__":
    main()