# Copyright 2023 Databricks, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import click
import numpy as np
import datasets
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_WITH_INPUT_KEYPATCH_FORMAT,
    PROMPT_WITH_INPUT_KEYWORD_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
)
from .mimiccxr_vq_dataset import (
    MimicCxrVqDataset, 
    sample_cxr_vq_input_instruction,
    sample_cxr_vq_output_instruction_with_position,
    sample_cxr_vq_output_instruction,
    get_inject_vq_fun,
    CXR_VQ_VQ_REPLACE_TEMPLATE,
    CXR_VQ_VQ_KEYPATCH_REPLACE_TEMPLATE,
    CXR_VQ_CODE_BOOK_SIZE, 
    CXR_VQ_VQ_LEN
)
from .openI_dataset import OpenIDataset

from transformers import BatchEncoding
import transformers.data.data_collator as data_collator


logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).parent.parent

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    
    def _torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            # for item in examples:
            #     print('input_ids:',len(item['input_ids']),'attention_mask:',len(item['attention_mask']))
            # try:
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of, padding=True)
            # except:
            #     for item in examples:
            #         print('input_ids:',len(item['input_ids']),'attention_mask:',len(item['attention_mask']))
            # print("------after------")
            # print('after --- input_ids:',batch['input_ids'].shape)
            # print('after --- attention_mask:', batch['attention_mask'].shape)

        else:
            batch = {
                "input_ids": data_collator._torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # batch = super().torch_call(examples)
        batch = self._torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels
        # print('batch["input_ids"]:',batch["input_ids"].shape)
        return batch


def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_batch_mimiccxrvq(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int, inject_vq_fun, use_matched_patches, topk_patch) -> dict:
    tokenizer_output = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )
    if use_matched_patches:
        # input_idss = [inject_vq_fun(ii, vq, p_vq) for ii, vq, p_vq in zip(tokenizer_output["input_ids"], batch['cxr_vq_shifted'], batch['patch_vq'])]
        # attention_masks = [[1 for _ in range(len(am) + CXR_VQ_VQ_LEN - 1)] for am in tokenizer_output["attention_mask"]] #  + topk_patch
        input_idss = []
        attention_masks = []
        for ii, vq, p_vq, am in zip(tokenizer_output["input_ids"], batch['cxr_vq_shifted'], batch['patch_vq'], tokenizer_output["attention_mask"]):
            input_idss_out, first_dx = inject_vq_fun(ii, vq, p_vq)
            input_idss.append(input_idss_out)
            am_len = ( CXR_VQ_VQ_LEN - 1) if first_dx is None else ( CXR_VQ_VQ_LEN + len(p_vq) - 3 )
            attention_masks.append([1 for _ in range(len(am) + am_len)])
            assert len(input_idss_out) == len(attention_masks[-1]), "the lengthes are not equal. input_idss:{}, attention_masks:{}".format(len(input_idss_out), len(attention_masks[-1]))
            # if len(input_idss_out) != len(attention_masks[-1]):
            #     print("the lengthes are not equal. input_idss:{}, attention_masks:{}".format(len(input_idss_out), len(attention_masks[-1])))
            #     if p_vq is not None:
            #         print("before-input_ids:",len(ii),"after-input_ids:",len(input_idss_out),"first_dx:",first_dx, len(attention_masks[-1]),'patch_vq:',len(p_vq))
            #     else:
            #         print("before-input_ids:",len(ii),"after-input_ids:",len(input_idss_out),"first_dx:",first_dx, len(attention_masks[-1]),'patch_vq:',p_vq)
            #     # assert len(input_idss) == len(attention_masks), "the lengthes are not equal. input_idss:{}, attention_masks:{}".format(len(input_idss), len(attention_masks))
            # print("before-input_ids:",len(ii),"after-input_ids:",len(input_idss_out),"first_dx:",first_dx, len(attention_masks[-1]),'patch_vq:',len(p_vq))
    else:
        input_idss = [inject_vq_fun(ii, vq) for ii, vq in zip(tokenizer_output["input_ids"], batch['cxr_vq_shifted'])]
        attention_masks = [[1 for _ in range(len(am) + CXR_VQ_VQ_LEN - 1)] for am in tokenizer_output["attention_mask"]]

    batch["input_ids"] = input_idss
    batch["attention_mask"] = attention_masks

    return batch


def load_training_dataset(path_or_dataset: str = "./data/databricks-dolly-15k.jsonl") -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset("json", data_files=path_or_dataset)["train"]
    logger.info("Found %d rows", dataset.num_rows)

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        # For some instructions there is an input that goes along with the instruction, providing context for the
        # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
        # some piece of information from it.  The response is that information to extract.  In other cases there is
        # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
        # born.
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
        return rec

    dataset = dataset.map(_add_text)

    return dataset


def load_training_dataset_mimiccxrvq(tokenizer_len, stage, only_use_findings, use_origin_report, dataset_name, \
                                        use_keywords, keywords_path, key_topk, \
                                        use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position) -> Dataset:
    if dataset_name == 'mimic-cxr':
        logger.info(f"Loading dataset from MIMIC-CXR-VQ-Dataset")
        dataset_train = MimicCxrVqDataset("path_to_mimic_cxr_raw_data/mimic-cxr-reports", 
                                        "../material/anno/mimiccxr_vqgan1024_res256_3e_codebook_indices.pickle", 
                                        tokenizer_len, 
                                        "train",
                                        stage,
                                        "../material/anno",
                                        only_use_findings, use_origin_report, use_keywords, keywords_path, key_topk, use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)
        dataset_test  = MimicCxrVqDataset("path_to_mimic_cxr_raw_data/mimic-cxr-reports", 
                                        "../material/anno/mimiccxr_vqgan1024_res256_3e_codebook_indices.pickle", 
                                        tokenizer_len, 
                                        "test",
                                        stage,
                                        "../material/anno",
                                        only_use_findings, use_origin_report, use_keywords, keywords_path, key_topk, use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)
    elif dataset_name == 'openI':
        logger.info(f"Loading dataset from OpenI-Dataset")
        codebook_path = "path/openI_vqgan1024_codebook_all.pickle"
        dataset_train = OpenIDataset("", 
                                        codebook_path, 
                                        tokenizer_len, 
                                        "train",
                                        stage,
                                        "data_root/dataset/x_ray/openI/anno/",
                                        only_use_findings, use_origin_report, use_keywords, keywords_path, key_topk, use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)
        dataset_test  = OpenIDataset("", 
                                        codebook_path, 
                                        tokenizer_len, 
                                        "test",
                                        stage,
                                        "data_root/dataset/x_ray/openI/anno/",
                                        only_use_findings, use_origin_report, use_keywords, keywords_path, key_topk, use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)
    else:
        print("Please give a correct dataset name: mimic-cxr or openI.")
        exit(0)

    
    dataset_train = Dataset.from_list(dataset_train)
    dataset_test = Dataset.from_list(dataset_test)

    logger.info("Found %d/%d rows", dataset_train.num_rows, dataset_test.num_rows)

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
            # print('rec["text"]:',rec["text"])
        elif rec['io_type'] == 'output':
            output_instruction = sample_cxr_vq_output_instruction_with_position(rec["view_pos"]) if ("view_pos" in rec and rec["view_pos"] is not None) else sample_cxr_vq_output_instruction()
            if rec["patch_vq"] is not None:
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
    remove_columns = ["report"]
    if use_keywords:
        remove_columns.append("keyword")
    if use_view_position:
        remove_columns.append("view_pos")

    # remove_columns = ["report", "keyword"] if use_keywords else "report"

    dataset_train = dataset_train.map(_add_text, remove_columns=remove_columns)
    dataset_test = dataset_test.map(_add_text, remove_columns=remove_columns)

    return dataset_train, dataset_test


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    model.resize_token_embeddings(len(tokenizer) + CXR_VQ_CODE_BOOK_SIZE)

    return model, tokenizer


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset()

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)
    logger.info("Done preprocessing")

    return dataset


def preprocess_dataset_mimiccxrvq(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED, stage=None, only_use_findings=False, use_origin_report=False, dataset_name='mimic-cxr', \
                                    use_keywords=False, keywords_path="", key_topk=10, use_matched_patches=False, patches_path="", keypatch_topk=5, use_it=False, use_view_position=False) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset_train, dataset_test = load_training_dataset_mimiccxrvq(len(tokenizer), stage, only_use_findings, use_origin_report, dataset_name, \
                                                                        use_keywords, keywords_path, key_topk, \
                                                                            use_matched_patches, patches_path, keypatch_topk, use_it, use_view_position)

    logger.info("Preprocessing dataset MIMIC-CXR-VQ")
    _preprocessing_function = partial(preprocess_batch_mimiccxrvq, 
                                      max_length=max_length, 
                                      tokenizer=tokenizer, 
                                      inject_vq_fun=get_inject_vq_fun(tokenizer, use_matched_patches, keypatch_topk),
                                      use_matched_patches=use_matched_patches,
                                      topk_patch=keypatch_topk)
    remove_columns = ["cxr_vq_shifted", "text", "io_type"]
    if use_matched_patches:
        remove_columns.append("patch_vq")
    
    dataset_train = dataset_train.map(
        _preprocessing_function,
        batched=True,
        remove_columns=remove_columns,
    )
    dataset_test = dataset_test.map(
        _preprocessing_function,
        batched=True,
        remove_columns=remove_columns,
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset MIMIC-CXR-VQ has %d/%d rows", dataset_train.num_rows, dataset_test.num_rows)
    dataset_train = dataset_train.filter(lambda rec: len(rec["input_ids"]) < max_length)
    dataset_test = dataset_test.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset MIMIC-CXR-VQ has %d/%d rows after filtering for truncated records", dataset_train.num_rows, dataset_test.num_rows)
    logger.info("Done preprocessing MIMIC-CXR_VQ")

    return dataset_train, dataset_test


def train(
    *,
    input_model: str,
    local_output_dir: str,
    dbfs_output_dir: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    deepspeed: str,
    gradient_checkpointing: bool,
    local_rank: str,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    test_size: Union[float, int],
    save_total_limit: int,
    warmup_steps: int,
    stage: str,
    only_use_findings: bool,
    use_origin_report: bool,
    dataset_name: str,
    use_keywords: bool,
    keywords_path: str, 
    key_topk: int,
    use_matched_patches: bool,
    patches_path: str,
    keypatch_topk: int, 
    use_it: bool,
    use_view_position: bool
):

    set_seed(seed)

    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=input_model, gradient_checkpointing=gradient_checkpointing
    )

    # Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
    # The configuraton for the length can be stored under different names depending on the model.  Here we attempt
    # a few possible names we've encountered.
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")
    
    # only_use_findings = True
    split_dataset_mimiccxrvq_train, split_dataset_mimiccxrvq_test = preprocess_dataset_mimiccxrvq(tokenizer=tokenizer, max_length=max_length, seed=seed, stage=stage, \
                                                                                                    only_use_findings=only_use_findings, use_origin_report=use_origin_report, \
                                                                                                    dataset_name=dataset_name, use_keywords=use_keywords, keywords_path=keywords_path, key_topk=key_topk, \
                                                                                                    use_matched_patches=use_matched_patches, patches_path=patches_path, keypatch_topk=keypatch_topk, use_it=use_it, \
                                                                                                    use_view_position=use_view_position)
   
    # processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed)
    # split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)
    # split_dataset_train = split_dataset["train"]
    # split_dataset_test = split_dataset["test"]

    # merged_dataset_train = datasets.interleave_datasets([split_dataset_mimiccxrvq_train, split_dataset_train], [0.8, 0.2], seed=seed, stopping_strategy="all_exhausted")
    # merged_dataset_test = datasets.interleave_datasets([split_dataset_mimiccxrvq_test, split_dataset_test], [0.8, 0.2], seed=seed, stopping_strategy="all_exhausted")
    merged_dataset_train = split_dataset_mimiccxrvq_train
    merged_dataset_test = split_dataset_mimiccxrvq_test

    logger.info("Shuffling merged dataset")
    merged_dataset_train = merged_dataset_train.shuffle(seed=seed)
    merged_dataset_test = merged_dataset_test.shuffle(seed=seed)

    logger.info("Merged train data size: %d", merged_dataset_train.num_rows)
    logger.info("Merged test data size: %d", merged_dataset_test.num_rows)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    if not dbfs_output_dir:
        logger.warn("Will NOT save to DBFS")

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=False,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="epoch",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=False,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=merged_dataset_train,
        eval_dataset=merged_dataset_test,
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    if dbfs_output_dir:
        logger.info(f"Saving Model to {dbfs_output_dir}")
        trainer.save_model(output_dir=dbfs_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--input-model", type=str, help="Input model to fine tune", default=DEFAULT_INPUT_MODEL)
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option(
    "--test-size", type=int, default=1000, help="Number of test records for evaluation, or ratio of test records."
)
@click.option("--warmup-steps", type=int, default=None, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=50, help="How often to run evaluation on test records")
@click.option("--save-steps", type=int, default=400, help="How often to checkpoint the model")
@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
@click.option("--stage", type=int, help="Training stage to run.")
@click.option("--only-use-findings", type=bool, default=False, help="Whether to only use findings as report.")
@click.option("--use-origin-report", type=bool, default=False, help="Whether to use origin report.")
@click.option("--dataset-name", type=str, default="mimic-cxr", help="Dataset name: mimic-cxr or openI")
# keywords
@click.option("--use-keywords", type=bool, default=False, help="Whether to use keywords.")
@click.option("--keywords-path", type=str, default="", help="path for keywords pickle.")
@click.option("--key-topk", type=int, default=10, help="topk keywords to use.")
# matched patches
@click.option("--use-matched-patches", type=bool, default=False, help="Whether to use matched patches.")
@click.option("--patches-path", type=str, default="", help="path for matched patches.")

@click.option("--keypatch-topk", type=int, default=5, help="topk keypatches to use.")
@click.option("--use-it", type=bool, default=False, help="Whether to use it.")

@click.option("--use-view-position", type=bool, default=False, help="Whether to use view position.")

def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
