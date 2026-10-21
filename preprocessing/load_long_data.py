import numpy as np
import os
import datasets
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm


def length_filter(sample, tokenizer, text_field, min_length, max_length):
    length = len(tokenizer(sample[text_field])["input_ids"])
    return length, (length > min_length) & (length < max_length)


def load_long_data(dataset_name, dataset_subset, dataset_split, tokenizer, initial_text_key,
                   min_text_len, max_text_len, max_steps, num_samples, save_path, special_key):
    if "redpajama" in dataset_name.lower():
        d_test = datasets.load_dataset(dataset_name,
                                       languages=["en"], name="default",
                                       split=dataset_split, streaming=True)
    else:
        d_test = datasets.load_dataset(dataset_name, dataset_subset,
                                       split=dataset_split, streaming=True)
    buf_data_dict = {"context": [], "length": []}
    if special_key is not None:
        buf_data_dict.update({special_key: []})
    data_len = max_steps
    for idx, sample in tqdm(enumerate(iter(d_test))):
        curr_len, use_for_sample = length_filter(sample, tokenizer, initial_text_key, min_text_len, max_text_len)
        if use_for_sample:
            buf_data_dict["context"].append(sample[initial_text_key])
            buf_data_dict["length"].append(curr_len)
            if special_key is not None:
                buf_data_dict[special_key].append(sample[special_key])
        if idx % 10000 == 0:
            print("Number of samples on current step:", len(buf_data_dict["context"]))
        if len(buf_data_dict["context"]) > num_samples or idx > data_len:
            break
        if len(buf_data_dict["context"]) % 100000 == 0:
            os.makedirs(save_path, exist_ok=True)
            subsampled_df = datasets.Dataset.from_dict(buf_data_dict)
            subsampled_df.save_to_disk(save_path)
    os.makedirs(save_path, exist_ok=True)
    subsampled_df = datasets.Dataset.from_dict(buf_data_dict)
    subsampled_df.save_to_disk(save_path)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    load_long_data(args.dataset_name, args.dataset_subset, args.dataset_split,
                   tokenizer, args.initial_text_key, args.min_text_len, args.max_text_len, args.max_steps,
                   args.num_samples, args.save_path, args.special_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="monology/pile-uncopyrighted", help='input dataset for loading')
    parser.add_argument('--dataset_subset', type=str, default=None, help='input dataset for loading')
    parser.add_argument('--dataset_split', type=str, default=None, help='input dataset for loading')
    parser.add_argument('--tokenizer_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help='Tokenizer name from HF')
    parser.add_argument('--min_text_len', type=int, default=2048, help='min length of texts in tokens')
    parser.add_argument('--max_text_len', type=int, default=1e8, help='max length of texts in tokens')
    parser.add_argument('--max_steps', type=int, default=1e10, help='Max steps over the iterable dataset')
    parser.add_argument('--num_samples', type=int, default=15000, help='Number of filtered long texts')
    parser.add_argument('--save_path', type=str, default="../../data/long_context_raw/your_long_data")
    parser.add_argument('--initial_text_key', type=str, default="text")
    parser.add_argument('--special_key', type=str, default=None, help="Additional key from original dataset to save in the resulting data")
    args = parser.parse_args()
    main(args)
