import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
sys.path.append('..')
device = 'cpu'
import argparse
from functools import partial
from transformers import EarlyStoppingCallback
from trl import SFTTrainer
from lm_experiments_tools.instruction_utils import formatting_func_dispatcher, formatting_func_dispatcher_non_it


def build_model(model_name, device, use_peft=False):
    """ Load model"""
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map=device, 
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2", 
                                                 trust_remote_code=True,)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if model_name == "meta-llama/Llama-3.2-1B":
        # hotfix - load 8b instruct tokenizer and reuse chat template from instruct tokenizer
        it_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    if tokenizer.pad_token is None:
        print("PAD token is not set; using EOS token!")
        tokenizer.pad_token = tokenizer.eos_token
    if use_peft:
        print(model)
        lora_config = LoraConfig(
            r=16,
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
        return lora_model, tokenizer
    return model, tokenizer


def main(args):
    set_seed(args.seed)
    # FOR THIS MODEL -- No chat template applied
    model_name = args.model_name
    device = "cuda"
    DATASET_NAME = args.dataset
    SAVE_DIR = f"../{args.output_dir}/{model_name}/{DATASET_NAME.split('/')[-1]}_ift_{args.max_seq_length}_seed_{args.seed}"
    NUM_STEPS = args.num_steps

    model, tokenizer = build_model(model_name, device, use_peft=args.use_peft)
    try:
        dataset = datasets.load_dataset(DATASET_NAME)
    except:
        dataset = datasets.load_from_disk(DATASET_NAME)
    if "LongAlign" in DATASET_NAME:
        dataset["train"] = dataset["train"].filter(lambda example: example["length"] < args.max_seq_length)
    if "long_dolly" in DATASET_NAME:
        lens = [1024, 2048, 4096, 8192, 16384]
        percentages = np.sum(lens) / np.array(lens)
        percentages /= np.sum(percentages)
        joined_datasets = []
        for curr_len, perc in zip(lens, percentages):
            buf_dname = "_".join(DATASET_NAME.split("_")[:-1]) + f"_{curr_len}"
            dataset = datasets.load_from_disk(buf_dname)
            indices = np.random.choice(np.arange(len(dataset)), int(len(dataset) * perc), replace=False)
            joined_datasets.append(dataset.select(indices))
        data_dict = {"train": datasets.concatenate_datasets(joined_datasets)}
        dataset = datasets.DatasetDict(data_dict)
    if "alpaca" in DATASET_NAME:
        dataset["train"] = dataset["train"].rename_column("input", "context")
        dataset["train"] = dataset["train"].rename_column("output", "response")

    # format datasets and split on train/test
    if args.instruct:
        formatting_func = partial(formatting_func_dispatcher(DATASET_NAME), tokenizer=tokenizer)
    else:
        formatting_func = formatting_func_dispatcher_non_it
    formatted_dataset = dataset.map(formatting_func)
    #tokenizer.pad_token = tokenizer.eos_token
    # use default left padding
    #tokenizer.padding_side = "right"
    indices = np.arange(len(formatted_dataset["train"]))
    # in case of flan -- use only 1% for test
    if "flan" in DATASET_NAME:
        train_ids, test_ids = train_test_split(indices, test_size=0.01, train_size=0.99, random_state=42)
    else:
        train_ids, test_ids = train_test_split(indices, test_size=0.1, train_size=0.9, random_state=42)
    splits = ["train", "test"]
    all_indices = [train_ids, test_ids]
    data_dict = {}
    for split, indices in zip(splits, all_indices):
        data_dict[split] = formatted_dataset["train"].select(indices)
    formatted_dataset = datasets.DatasetDict(data_dict)
    print(formatted_dataset)
    # run instruction tuning
    supervised_finetuning_trainer = SFTTrainer(
        model,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=NUM_STEPS,
            evaluation_strategy="steps",
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            output_dir=SAVE_DIR,
            save_safetensors=False,
            seed=args.seed,
            load_best_model_at_end=args.early_stopping_tolerance!=-1
        ),
        packing=True,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    if args.early_stopping_tolerance != -1:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_tolerance
        )
        supervised_finetuning_trainer.add_callback(early_stopping)
    supervised_finetuning_trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default="glkuzi/flan_collection", help='dataset name')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B", help='model name')
    parser.add_argument('--instruct', type=int, default=0, help='use instruct template or not')
    parser.add_argument('--use_peft', type=int, default=1, help='use peft or not')
    parser.add_argument('--output_dir', type=str, default="qwen_exps")
    # train parameters
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--early_stopping_tolerance', type=int, default=-1)
    parser.add_argument('--max_seq_length', type=int, default=1024)

    args = parser.parse_args()
    main(args)
