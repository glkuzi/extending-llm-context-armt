# -*- coding: utf-8 -*-

# ! pip install datasets transformers
# ! git clone https://github.com/booydar/t5-experiments/ github

import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import sys
import tqdm
import torch
import datasets
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import set_seed
from itertools import chain
from torch.utils.data import DataLoader#, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# sys.path.append('github')
sys.path.append('..')
device = 'cpu'

from trl import SFTTrainer, SFTConfig

"""### Load model"""

set_seed(42)

model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# small workaround to properly create RMT from model
device = "cuda:0"
model.to(device)

"""## Prepare data from LongAlign"""
DATASET_NAME = "../data/flan_collection_seed_43"
SAVE_DIR = "../test_exps/gpt_2_ift_flan_1024_seed_42"
NUM_STEPS = 100000


dbricks_15k_dataset_base = datasets.load_dataset(DATASET_NAME)

def formatting_func(example):
  if example.get("context", "") != "":
      input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n"
      f"{example['response']}")

  else:
    input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response:\n"
      f"{example['response']}")

  return {"text" : input_prompt}

def formatting_func(example):
  input_prompt = (f"{example['inputs']}\n\n"
      f"{example['targets']}")

  return {"text" : input_prompt}


formatted_dataset = dbricks_15k_dataset_base.map(formatting_func)
tokenizer.pad_token = tokenizer.eos_token
# use default left padding
#tokenizer.padding_side = "right"

indices = np.arange(len(formatted_dataset["train"]))
train_ids, test_ids = train_test_split(indices, test_size=0.1, train_size=0.9, random_state=42)
splits = ["train", "test"]
all_indices = [train_ids, test_ids]
data_dict = {}
for split, indices in zip(splits, all_indices):
    data_dict[split] = formatted_dataset["train"].select(indices)
formatted_dataset = datasets.DatasetDict(data_dict)
print(formatted_dataset)

supervised_finetuning_trainer = SFTTrainer(
    model,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    args=TrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        learning_rate=5e-4,
        max_steps=NUM_STEPS,
        eval_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        output_dir=SAVE_DIR,
        save_safetensors=False,
        seed=42,
    ),
    packing=True,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,
)

supervised_finetuning_trainer.train()





"""

dbricks_15k_dataset_base = datasets.load_dataset("mosaicml/dolly_hhrlhf")

def formatting_func(example):
    return {"text": example["prompt"] + example['response']}

formatted_dataset = dbricks_15k_dataset_base.map(formatting_func)

formatted_dataset["train"][0]

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#indices = np.arange(len(formatted_dataset["train"]))
#train_ids, test_ids = train_test_split(indices, test_size=0.1, train_size=0.9, random_state=42)
#splits = ["train", "test"]
#all_indices = [train_ids, test_ids]
#data_dict = {}
#for split, indices in zip(splits, all_indices):
#    data_dict[split] = formatted_dataset["train"].select(indices)
#formatted_dataset = datasets.DatasetDict(data_dict)

supervised_finetuning_trainer = SFTTrainer(
    model,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    args=TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=10000,
        output_dir="../test_exps/gpt_2_ift_dolly_seed_42",
        save_safetensors=False,
        seed=42,
    ),
    packing=True,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
)

supervised_finetuning_trainer.train()
"""




"""

task_name = 'THUDM/LongAlign-10k'
raw_datasets = datasets.load_dataset('THUDM/LongAlign-10k')
# filter seq longer than 16k
raw_datasets = raw_datasets.filter(lambda example: (example["length"] <= 16384))
# split on train and test
indices = np.arange(len(raw_datasets["train"]))
train_ids, val_ids = train_test_split(indices, test_size=0.25, train_size=0.75, random_state=42)
test_ids, val_ids = train_test_split(val_ids, test_size=0.5, train_size=0.5, random_state=42)
splits = ["train", "validation", "test"]
all_indices = [train_ids, val_ids, test_ids]
data_dict = {}
for split, indices in zip(splits, all_indices):
    data_dict[split] = raw_datasets["train"].select(indices)
raw_datasets = datasets.DatasetDict(data_dict)

raw_datasets



SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
USER_TEMPLATE = '<context>\n{context}\n</context>\n\nQuestion: {question}'
USER_TEMPLATE_ANS = '<context>\n{context}\n</context>\n\nQuestion: {question}\n\nAnswer: {answer}'
DEFAULT_TEMPLATE = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'
DEFAULT_TEMPLATE_ANS = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE_ANS}'

CUSTOM_SYSTEM_PROMPTS = {
    # https://github.com/dvlab-research/LongLoRA/blob/2345c6d030f61ac3a031906386a103a5b05e0e6f/inference.py#L18
    'LONGLORA_LLAMA2':
        'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. '
        'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
        'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
        'If a question does not make any sense, or is not factually coherent, explain why instead of answering '
        'something not correct. If you don\'t know the answer to a question, please don\'t share false information.'
}


def get_formatted_input(context, question, examples, instruction, post_prompt, template=DEFAULT_TEMPLATE):
    # pre_prompt - general instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    formatted_input = template.format(instruction=instruction, examples=examples, post_prompt=post_prompt,
                                      context=context.strip(), question=question)
    return formatted_input.strip()

def get_formatted_input_with_answer(context, question, examples, instruction, post_prompt, answer, template=DEFAULT_TEMPLATE_ANS):
    # pre_prompt - general instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    formatted_input = template.format(instruction=instruction, examples=examples, post_prompt=post_prompt,
                                      context=context.strip(), question=question, answer=answer)
    return formatted_input.strip()

def formating_func(example):
    formatted_input = get_formatted_input_with_answer(
        context = "\n\n".join(example["messages"][0]["content"].split("\n\n")[:-1]),
        question = example["messages"][0]["content"].split("\n\n")[-1],
        examples = "",
        instruction = 'I will give you context with the facts and a question. You need to answer the question based only on the information from the facts.',
        post_prompt = "",
        answer = example["messages"][1]["content"]
    )
    return {"text": formatted_input}



long_dat = raw_datasets.map(formating_func)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

supervised_finetuning_trainer = SFTTrainer(
    model,
    train_dataset=long_dat["train"],
    eval_dataset=long_dat["validation"],
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_steps=5000,
        output_dir="../test_exps/rmt_gpt_2_ift_LA_seed_42",
        save_safetensors=False,
        seed=42,
    ),
    packing=True,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=16384,
)

supervised_finetuning_trainer.train()

"""
