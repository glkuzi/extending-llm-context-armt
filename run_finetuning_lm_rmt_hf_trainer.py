import logging
import os
if 'LOCAL_RANK' in os.environ and 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']
elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ and 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

import gc
import copy
os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
from itertools import chain
from datetime import timedelta
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from functools import partial
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed, BitsAndBytesConfig, AutoModelForCausalLM, TrainerCallback
from lm_experiments_tools.dataset_preprocessing import load_and_preprocess_task, prepare_greedy_packing, load_and_preprocess_babilong, collate_fn_mixin, DatasetMix
from lm_experiments_tools.instruction_utils import mask_non_completion, mask_non_completion_multi

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import pdb
import json

import accelerate
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
# torch.cuda.set_device(hvd.local_rank())

def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = HfArgumentParser(SFTConfig)
parser.add_argument('--task_name', type=str, help="Task name, wikitext, ...")
parser.add_argument('--append_concat_token', action='store_true', default=False, help="Append concat token during packing")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--sample_size', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--test_sample_size', type=int, default=-1, help='input sequnce length for test (default: -1).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attentinon mask, '
                    'eval on last segment only', default=False)
parser.add_argument('--attend_to_previous_input', action='store_true', help='attend to the previous segment', default=False)
parser.add_argument('--use_length_filtering', action='store_true', help='filter samples longer than train len', default=False)
parser.add_argument('--reduce_eval', type=float, default=None, help='part of eval to use')
parser.add_argument('--subsample_train', type=float, default=1.0, help='subsample train ratio, from 0 to 1')
parser.add_argument('--no_packing', action='store_true', help='disable packing, add padding', default=False)
parser.add_argument('--greedy_packing', action='store_true', help='greedy packing', default=False)
parser.add_argument('--use_full_attn_mask', action='store_true', help='use full attn mask in greedy packing', default=False)
parser.add_argument('--padding_side', type=str, help='set padding side', default=False)
parser.add_argument('--truncate_before', type=int, default=0, help='truncate input before defined ids, for debug only')
parser.add_argument('--truncate_only_train', action='store_true', help='set truncate input before defined ids only for train, for debug only', default=False)


# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=none_or_str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')
parser.add_argument('--checkpoint', type=str, default=None, help='Full experiment checkpoint, used to resume training in SFTTrainer')

# Aydar # RMT args
parser.add_argument('--segment_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--loss_from_last_seg_only', action='store_true', default=False, help='take loss from last segment only')
parser.add_argument('--no_loss_from_first_segment', action='store_true', default=False, help='turn off loss from first segment')
parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers', default=None)
parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
                    help='reconstuction loss ratio in total loss')
# parser.add_argument('--segment_ordering', type=str,help='????', default='regular',
#                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
parser.add_argument('--use_truncated_backward', action='store_true', default=False,
                    help='whether to use RMT truncated bptt method in backward')
parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')
parser.add_argument('--load_optimizer', type=int, default=1, help='load optimizer')
parser.add_argument('--tune_only_memory', action='store_true', default=False,
                    help='Stop training all model weights except memory layer memory')
parser.add_argument('--tune_only_armt', action='store_true', default=False,
                    help='Stop training all model weights except ARMT params')
parser.add_argument('--no_tune_armt', action='store_true', default=False,
                    help='Stop training all model weights except ARMT params')
parser.add_argument('--mask_non_completion', action='store_true', default=False,
                    help='Mask everything except completion in dataset')
# ARMT parameters
parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')
parser.add_argument('--no_correction', action='store_true', default=False,
                    help='ARMT shmidhuber correction for rewriting')
parser.add_argument('--wrap_pos', action='store_true', default=False,
                    help='Wrap positional encoding for memory tokens (default: False)')
parser.add_argument('--prev_seg_kv', action='store_true', default=False, help='propagate kv from previous segment')
parser.add_argument('--use_sink', action='store_true', default=False, help='use_attention_sink_token')
parser.add_argument('--n_heads', type=int, default=None, help='number of heads in associative matrix')
parser.add_argument('--freeze_mem', action='store_true', default=False, help='freeze armt memory')
parser.add_argument('--compress_mem', type=int, default=0, help='compression size for memory mapping')
parser.add_argument('--identity_frequency', type=int, default=0, help='frequency of identity associative layers')
parser.add_argument('--non_identity_layers', type=str, default=None, help='indices of non-identity associative layers in ARMT')

# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')
parser.add_argument('--tokenizer_for_chat_template', type=str, default=None, help='path or name of pre-trained HF Tokenizer, from which CT are used')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')
parser.add_argument('--early_stopping_patience', type=int, default=-1,
                    help='Early stopping tolerance')

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--use_qlora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')
parser.add_argument('--lora_layers', type=str, default=None, help='')
parser.add_argument('--add_lora_to_armt', action='store_true', default=False, help='')
parser.add_argument('--add_full_lora_to_armt', action='store_true', default=False, help='')
parser.add_argument('--merge_lora_checkpoint', action='store_true', default=False, help='')

# Parallel Adapter args
parser.add_argument('--use_adapter', action='store_true', default=False, help='')
parser.add_argument('--adapter_bottleneck_dim', type=int, default=512, help='')
parser.add_argument('--adapter_dropout', type=float, default=0.1, help='')
parser.add_argument('--adapter_scale', type=float, default=4.0, help='')

parser.add_argument('--distillator_cls', type=str, default=None, help='distillator class for RMT')
parser.add_argument('--teacher_cls', type=str, default=None, help='teacher class')
parser.add_argument('--alpha_distil', type=float, default=None, help='')
parser.add_argument('--pretrained_teacher', type=str, help='teacher model name in HF Model Hub (default: "")')
# Babilong params
parser.add_argument('--mixed_length_ratio', type=float, default=0.0, help='used for mixed length curriculum. '
                    'r > 0.0 means that we will start to sample batches with lengths <= max_n_segments')
parser.add_argument('--max_n_facts', type=int, default=None, help='drop samples with higher number of facts')
parser.add_argument('--task_start_pct', type=float, default=None, help='left border of facts in sample, between 0 and 1')
parser.add_argument('--task_end_pct', type=float, default=None, help='right border of facts in sample, between task_start_pct and 1')
parser.add_argument('--no_noise', action='store_true', default=False, help='turn off noise in babilong')
parser.add_argument('--task_dataset', type=str, help="Task name", default="qa1_single-supporting-fact")
parser.add_argument('--test_task_dataset', type=str, help="Task name", default=None)
parser.add_argument('--noise_dataset', type=str, help="Task name", default='wikitext')
parser.add_argument('--noise_dataset_split', type=str, help="Task name", default=None)
parser.add_argument('--babi_path', type=str, help="path to babi folder", default="data/tasks_1-20_v1-2/en-10k")
parser.add_argument('--add_chat_template', action='store_true', default=False, help='use chat template instead base formatting')
parser.add_argument('--mask_type', type=str, default="base", help='label mask for babilong')
parser.add_argument('--shift_mask_left', action='store_true', default=False, help='shift labels mask left')
parser.add_argument('--mixing_dataset_ratio', type=float, default=0.5, help='mixing ratio for BL and inst data')
# yarn params
parser.add_argument('--use_yarn', action='store_true', default=False,
                    help='Train base model with YARN context extension; only for Llama-like')
parser.add_argument('--yarn_original_max_position_embeddings', type=int, default=8192, help='')
parser.add_argument('--yarn_scaling_type', type=str, default="yarn")
parser.add_argument('--yarn_scaling_factor', type=float, default=16.0)
parser.add_argument('--yarn_rope_theta', type=float, default=100000.0)
parser.add_argument('--yarn_sliding_window_attention_schedule', type=str)
parser.add_argument('--yarn_max_position_embeddings', type=int)
parser.add_argument('--use_yarn_hf', action='store_true', default=False,
                    help='Train base model with YARN context extension in hf version; only for Llama-like')
# new armt params
parser.add_argument('--use_armt_pretrain', type=str,
                    help='Train ARMT with inner implementation from LM pretrain')
parser.add_argument('--model_cpt_v2', type=none_or_str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--remove_assoc_layer', type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    set_seed(args.seed)

    # workaround with setting bigger tiomeout for NCCL (useful for big dataset, to avoid timeout at tokenization)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[accelerate.InitProcessGroupKwargs(timeout=timedelta(seconds=20 * 1800))])
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.output_dir is None:
        logger.warning('output_dir is not set: config, logs and checkpoints will not be saved.')

    if not args.from_pretrained or args.merge_lora_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    if args.tokenizer_for_chat_template is not None:
        it_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_for_chat_template, trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
        tokenizer_name = args.tokenizer_for_chat_template
    else:
        tokenizer_name = args.from_pretrained
    padding_side = "right"
    if args.padding_side:
        tokenizer.padding_side = args.padding_side
        padding_side = args.padding_side
    # Prepare datasets
    logger.info(f'preparing dataset for {args.task_name}')
    test_task_dataset = args.task_dataset if args.test_task_dataset is None else args.test_task_dataset
    with accelerator.main_process_first():
        if "babilong" in args.task_name:
            test_sample_size = args.sample_size if args.test_sample_size == -1 else args.test_sample_size
            tokenized_datasets, collate_fn_babilong_wrapped = load_and_preprocess_babilong(tokenizer, args.sample_size,
                                                                                           test_sample_size,
                                                                                           args.max_n_segments,
                                                                                           args.noise_dataset, args.noise_dataset_split,
                                                                                           args.babi_path, args.task_dataset,
                                                                                           test_task_dataset,
                                                                                           args.max_n_facts,
                                                                                           args.vary_n_segments, args.segment_size,
                                                                                           args.task_start_pct, args.task_end_pct,
                                                                                           args.mixed_length_ratio,
                                                                                           args.add_chat_template,
                                                                                           args.mask_type,
                                                                                           args.num_mem_tokens)
            if "babilong+" in args.task_name:
                # load additional dataset, and mix it with babilong
                secondary_task_name = args.task_name.split("+")[1]
                tokenized_datasets_second = load_and_preprocess_task(secondary_task_name, tokenizer, args.sample_size,
                                                              args.max_n_segments, args.num_mem_tokens,
                                                              args.use_length_filtering, args.reduce_eval,
                                                              args.subsample_train)
        else:
            tokenized_datasets = load_and_preprocess_task(args.task_name, tokenizer, args.sample_size,
                                                          args.max_n_segments, args.num_mem_tokens,
                                                          args.use_length_filtering, args.reduce_eval,
                                                          args.subsample_train)
    block_size = args.segment_size
    # TODO: add a switch for ARMT
    if args.num_mem_tokens is not None:
        block_size -= 2 * args.num_mem_tokens
        if args.d_mem is not None:
            # cause for ARMT we have mem tokens only at the end
            block_size = args.segment_size - args.num_mem_tokens
    logger.info(f'Set block size to {block_size}')
    history_size = args.sample_size - block_size

    # Check grouping in chunks, compare with TRL packing
    # It's almost the same, but maybe we need an additional separator?
    def pad_last_seq(seq, block_size, key):
        if key in ["input_ids", "labels"]:
            padding_value = tokenizer.pad_token_id
        else:
            padding_value = 0
        if len(seq) < block_size:
            seq = seq + [padding_value] * (block_size - len(seq))
        return seq
    def group_texts(examples, block_size, history_size=None):
        if args.mask_non_completion:
            if "llama" in args.from_pretrained.lower() or "llama" in tokenizer_name.lower():
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
                response_template_end_idx = 128009
            elif "qwen" in args.from_pretrained.lower() or "qwen" in tokenizer_name.lower():
                response_template = "<|im_start|>assistant\n"
                response_template_end_idx = 151645
            elif "smollm2" in args.from_pretrained.lower() or "smollm2" in tokenizer_name.lower():
                response_template = "<|im_start|>assistant"
                response_template_end_idx = 2
            elif "gemma" in args.from_pretrained.lower() or "gemma" in tokenizer_name.lower():
                #response_template = "<end_of_turn>\n<start_of_turn>model\n"
                response_template = "<start_of_turn>model\n"
                response_template_end_idx = 106
            else:
                raise NotImplementedError("For this type of model or tokenizer non-completion masking is not supported")
            examples["labels"] = [mask_non_completion_multi(el, response_template, response_template_end_idx, tokenizer, mask_type=args.mask_type) for el in examples["input_ids"]]
        if args.append_concat_token:
            examples["input_ids"] = [el + [tokenizer.eos_token_id] for el in examples["input_ids"]]
            if "labels" in examples.keys():
                examples["labels"] = [el + [tokenizer.eos_token_id] for el in examples["labels"]]
            if "attention_mask" in examples.keys():
                examples["attention_mask"] = [el + [1] for el in examples["attention_mask"]]
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if history_size is None:
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        else:
            result = {
                k: [pad_last_seq(t[max({0, i - history_size}) : i + block_size], history_size + block_size, k) for i in range(history_size, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            for i in range(len(result['input_ids'])):
                if len(result['input_ids'][i]) != history_size + block_size:
                    print(total_length, len(result["input_ids"]))
                    print(concatenated_examples.keys())
                    print(i, len(result['input_ids'][i]), history_size, block_size)
                    assert False
        if not("labels" in examples.keys()):
            result["labels"] = result["input_ids"].copy()
        return result
    def prepare_texts(examples, truncate_before):
        if args.mask_non_completion:
            if "llama" in args.from_pretrained.lower() or "llama" in tokenizer_name.lower():
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
                response_template_end_idx = 128009
            elif "qwen" in args.from_pretrained.lower() or "qwen" in tokenizer_name.lower():
                response_template = "<|im_start|>assistant\n"
                response_template_end_idx = 151645
            elif "smollm2" in args.from_pretrained.lower() or "smollm2" in tokenizer_name.lower():
                response_template = "<|im_start|>assistant"
                response_template_end_idx = 2
            elif "gemma" in args.from_pretrained.lower() or "gemma" in tokenizer_name.lower():
                #response_template = "<end_of_turn>\n<start_of_turn>model\n"
                response_template = "<start_of_turn>model\n"
                response_template_end_idx = 106
            else:
                raise NotImplementedError("For this type of model or tokenizer non-completion masking is not supported")
            examples["labels"] = [mask_non_completion_multi(el, response_template, response_template_end_idx, tokenizer, mask_type=args.mask_type) for el in examples["input_ids"]]
        #print(examples["labels"][0])
        result = {
            k: [el[truncate_before:] for el in t]
            for k, t in examples.items()
        }
        if not("labels" in examples.keys()):
            result["labels"] = result["input_ids"].copy()
        return result

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.sliding_window:
        def collate_fn(batch):
            # deprecated
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            input_lens = [el.shape[-1] for el in input_ids]

            labels = [torch.tensor(b['labels']) for b in batch]
            attention_mask = [torch.tensor(b['attention_mask']) for b in batch]
            input_ids = pad_sequence(input_ids, padding_value=id_pad_value, padding_side=padding_side).T
            labels = pad_sequence(labels, padding_value=-100, padding_side=padding_side).T
            attention_mask = pad_sequence(attention_mask, padding_value=0, padding_side=padding_side).T

            # make sliding window att mask
            attention_mask = attention_mask[:, None, :].repeat(1, attention_mask.shape[1], 1)
            attention_mask = (torch.tril(attention_mask, 0) * (1 - torch.tril(attention_mask, -block_size)))

            collated = {'input_ids': input_ids,
                        'labels': labels, 
                        'attention_mask': attention_mask}

            if input_ids.shape[1] != block_size and args.num_mem_tokens is not None:
                # take only labels for last block (maybe use all labels during training?)
                labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for i, lens in enumerate(input_lens):
                    labels_mask[i, max(lens - block_size, 0): lens] = True
                collated['labels_mask'] = labels_mask

            return collated
    else:
        def collate_fn(batch):
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            if not("labels" in batch[0].keys()):
                labels = [torch.tensor(b['input_ids'].copy()) for b in batch]
            else:
                labels = [torch.tensor(b['labels']) for b in batch]
            attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]


            labels_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
            
            if getattr(args, 'loss_from_last_seg_only', False):
                for m in labels_mask:
                    m[:-args.segment_size] = False

            if getattr(args, 'no_loss_from_first_segment', False):
                for m in labels_mask:
                    m[:args.segment_size] = False

            input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True, padding_side=padding_side)
            labels = pad_sequence(labels, padding_value=-100, batch_first=True, padding_side=padding_side)
            attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True, padding_side=padding_side)
            labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True, padding_side=padding_side)

            collated = {'input_ids': input_ids,
                        'labels': labels, 
                        'attention_mask': attention_mask,
                        }
            if args.num_mem_tokens is not None:
                # add labels mask only for RMT, ARMT
                collated['labels_mask'] = labels_mask.bool()
            # TODO: add masking for response only
            return collated
    if not(args.no_packing) and not(args.greedy_packing):
        with accelerator.main_process_first():
            train_dataset = tokenized_datasets["train"].map(lambda x: group_texts(x, block_size, history_size),
                                                            batched=True, desc=f"Grouping train in chunks of {block_size} and history {history_size}")
            valid_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, block_size, history_size), 
                                                                batched=True, desc=f"Grouping valid in chunks of {block_size}")
            test_dataset = tokenized_datasets["test"].map(lambda x: group_texts(x, block_size, history_size),
                                                          batched=True, desc=f"Grouping test in chunks of {block_size}")
    elif not(args.no_packing) and args.greedy_packing:
        with accelerator.main_process_first():
            train_dataset = tokenized_datasets["train"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                            batched=True, desc=f"Preparing train")
            if args.truncate_only_train:
                args.truncate_before = 0
            valid_dataset = tokenized_datasets["validation"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                                batched=True, desc=f"Preparing valid")
            test_dataset = tokenized_datasets["test"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                          batched=True, desc=f"Preparing test")
        train_dataset = prepare_greedy_packing(train_dataset, tokenizer, args.sample_size, not(args.use_full_attn_mask))
        valid_dataset = prepare_greedy_packing(valid_dataset, tokenizer, args.sample_size, not(args.use_full_attn_mask))
        test_dataset = prepare_greedy_packing(test_dataset, tokenizer, args.sample_size, not(args.use_full_attn_mask))
    else:
        with accelerator.main_process_first():
            """
            if args.task_name == "babilong":
                kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers, 'collate_fn': collate_fn_babilong_wrapped}
                per_worker_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                per_worker_eval_batch_size = int(np.ceil(args.per_device_train_batch_size // 2)) * args.gradient_accumulation_steps
                train_sampler = DistributedSampler(tokenized_datasets["train"], rank=accelerator.process_index,
                                                num_replicas=accelerator.num_processes, shuffle=True, drop_last=True,
                                                seed=args.seed)
                test_sampler = DistributedSampler(tokenized_datasets["test"], rank=accelerator.process_index,
                                                num_replicas=accelerator.num_processes, drop_last=False, shuffle=False)
                train_dataset = DataLoader(batch_size=per_worker_batch_size, dataset=tokenized_datasets["train"], sampler=train_sampler,
                                            **kwargs)
                valid_dataset = DataLoader(batch_size=per_worker_eval_batch_size, dataset=tokenized_datasets["test"], sampler=test_sampler, **kwargs)
                """
            if "babilong" in args.task_name:
                train_dataset = tokenized_datasets["train"]
                valid_dataset = tokenized_datasets["test"]
                print(train_dataset)
                if "babilong+" in args.task_name:
                    train_dataset_second = tokenized_datasets_second["train"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                                    batched=True, desc=f"Preparing train")
                    if args.truncate_only_train:
                        args.truncate_before = 0
                    valid_dataset_second = tokenized_datasets_second["validation"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                                        batched=True, desc=f"Preparing valid")
                    test_dataset_second = tokenized_datasets_second["test"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                                batched=True, desc=f"Preparing test")
                    train_dataset = DatasetMix([train_dataset, train_dataset_second], mixing_ratio=args.mixing_dataset_ratio)
                    valid_dataset = DatasetMix([valid_dataset, valid_dataset_second], mixing_ratio=args.mixing_dataset_ratio, is_train=False)
                #print(train_dataset.keys())
            else:
                train_dataset = tokenized_datasets["train"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                                batched=True, desc=f"Preparing train")
                if args.truncate_only_train:
                    args.truncate_before = 0
                valid_dataset = tokenized_datasets["validation"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                                    batched=True, desc=f"Preparing valid")
                test_dataset = tokenized_datasets["test"].map(lambda x: prepare_texts(x, args.truncate_before),
                                                            batched=True, desc=f"Preparing test")
            # TODO: add here greedy packing with completion-only loss
            #train_dataset = tokenized_datasets["train"]
            #valid_dataset = tokenized_datasets["validation"]
            #test_dataset = tokenized_datasets["test"]
    # define model
    # TODO: move model building to separate function
    model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')

    if args.use_adapter:
        model_cfg = AutoConfig.from_pretrained(args.from_pretrained)

        model_cfg.use_parallel_adapter = args.use_adapter
        model_cfg.parallel_adapter_mode = 'ffn'
        model_cfg.adapter_bottleneck_dim = args.adapter_bottleneck_dim
        model_cfg.adapter_dropout = args.adapter_dropout
        model_cfg.adapter_scale = args.adapter_scale

        model = model_cls(config=model_cfg)

        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        base_model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)

        model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f'Added adapters')
    else:
        # TODO: fix if for Qwen and Llama
        if not args.from_pretrained:
            model_cfg = AutoConfig.from_pretrained(args.model_cfg)
            model = model_cls(config=model_cfg)
        else:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
            if args.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_storage=torch.bfloat16,
                )
            if "Qwen" in args.from_pretrained or "Llama" in args.from_pretrained or "SmolLM" in args.from_pretrained or "gemma" in args.from_pretrained:
                attn_implementation = "flash_attention_2"
                if args.prev_seg_kv:
                    attn_implementation = "sdpa"
                if args.greedy_packing:
                    attn_implementation = "sdpa"
                if "gemma" in args.from_pretrained:
                    attn_implementation = "eager"
                if args.use_yarn:
                    from lm_experiments_tools.modeling_llama_yarn import LlamaForCausalLM
                    from lm_experiments_tools.configuration_llama import LlamaConfig
                    config_cls = LlamaConfig
                    model_cls = LlamaForCausalLM
                    original_max_position_embeddings = args.yarn_original_max_position_embeddings if args.yarn_original_max_position_embeddings else 8192
                    config = config_cls.from_pretrained(args.from_pretrained)
                    config.rope_scaling = {
                        "type": args.yarn_scaling_type,
                        "factor": args.yarn_scaling_factor,
                        "original_max_position_embeddings": original_max_position_embeddings
                    }
                    config.rope_theta = args.yarn_rope_theta
                    config.max_position_embeddings = int(args.yarn_scaling_factor * original_max_position_embeddings) \
                        if not args.yarn_max_position_embeddings else args.yarn_max_position_embeddings

                    sliding_window_attention_schedule = [int(x) for x in args.yarn_sliding_window_attention_schedule.split(",")] \
                        if args.yarn_sliding_window_attention_schedule else None
                    if sliding_window_attention_schedule is not None and len(sliding_window_attention_schedule) == 1:
                        config.sliding_window = sliding_window_attention_schedule[0]
                        accelerator.print(
                            f"Sliding attention window set to {config.sliding_window}")

                    model = model_cls.from_pretrained(
                        args.from_pretrained,
                        torch_dtype=torch.bfloat16,
                        config=config,
                        attn_implementation=attn_implementation,
                    )
                if args.use_yarn_hf:
                    from transformers import LlamaConfig
                    config_cls = LlamaConfig
                    config = config_cls.from_pretrained(args.from_pretrained)
                    original_max_position_embeddings = args.yarn_original_max_position_embeddings if args.yarn_original_max_position_embeddings else 8192
                    config.rope_parameters = {
                        "rope_type": "yarn",
                        "factor": args.yarn_scaling_factor,
                        "original_max_position_embeddings": original_max_position_embeddings,
                    }
                    model = model_cls.from_pretrained(
                        args.from_pretrained,
                        torch_dtype=torch.bfloat16,
                        config=config,
                        attn_implementation=attn_implementation,
                    )
                elif args.use_armt_pretrain:
                    model = AutoModelForCausalLM.from_pretrained(args.use_armt_pretrain,
                                                                 trust_remote_code=True)
                    if args.remove_assoc_layer is not None:
                        # take specific assoc. layer and replace forward with identity
                        import types
                        def id_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                            return hidden_states
                        def _associate_with_mem(self, hidden_states: torch.Tensor, W_mem: torch.Tensor, z):
                            return hidden_states
                        def _update_mem_with_mem(
                            self, mem_tokens: torch.Tensor, W_mem: torch.Tensor, z, first_seg,
                        ):
                            return W_mem, z, False
                        if ";" in args.remove_assoc_layer:
                            remove_assoc_layers = args.remove_assoc_layer.split(";")
                            remove_assoc_layers = [int(el) for el in remove_assoc_layers]
                            print(remove_assoc_layers)
                            for remove_assoc_layer in remove_assoc_layers:
                                model.model.model.layers[remove_assoc_layer]._associate_with_mem = types.MethodType(_associate_with_mem, model.model.model.layers[remove_assoc_layer])
                                model.model.model.layers[remove_assoc_layer]._update_mem_with_mem = types.MethodType(_update_mem_with_mem, model.model.model.layers[remove_assoc_layer])
                        else:
                            remove_assoc_layer = int(args.remove_assoc_layer)
                            print(model.model.model.layers[remove_assoc_layer])
                            model.model.model.layers[remove_assoc_layer]._associate_with_mem = types.MethodType(_associate_with_mem, model.model.model.layers[remove_assoc_layer])
                            model.model.model.layers[remove_assoc_layer]._update_mem_with_mem = types.MethodType(_update_mem_with_mem, model.model.model.layers[remove_assoc_layer])
                        #model.model.model.layers[args.remove_assoc_layer].forward = id_forward
                else:
                    model = model_cls.from_pretrained(args.from_pretrained,
                                                    attn_implementation=attn_implementation,
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=bnb_config if args.use_qlora else None,
                                                    #device_map="cuda",
                                                    trust_remote_code=True)
            else:
                model = model_cls.from_pretrained(args.from_pretrained,
                                                  quantization_config=bnb_config if args.use_qlora else None,
                                                  use_safetensors=False)
            if args.merge_lora_checkpoint:
                model = model_cls.from_pretrained(args.tokenizer,
                                                  attn_implementation=attn_implementation,
                                                  torch_dtype=torch.bfloat16,
                                                  quantization_config=bnb_config if args.use_qlora else None,
                                                  trust_remote_code=True)
                model = PeftModel.from_pretrained(model, args.from_pretrained)
                model.merge_and_unload()
                model = model.base_model.model
    if args.use_lora:
        if args.lora_layers == "all-linear" and args.use_armt_pretrain:
            exclude_armt = {"W_mq", "W_mk", "W_mv", "W_mb", "lm_head"}
            try:
                from transformers.pytorch_utils import Conv1D
            except Exception:
                Conv1D = ()
            lora_layers = sorted({
                name.split(".")[-1]
                for name, module in model.named_modules()
                if isinstance(module, (torch.nn.Linear, Conv1D))
                and name.split(".")[-1] not in exclude_armt
            })
        else:
            lora_layers = args.lora_layers.split(";") if ";" in args.lora_layers else args.lora_layers
        #lora_layers = args.lora_layers.split(";") if ";" in args.lora_layers else args.lora_layers
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_attn_dim,
            lora_alpha=args.lora_attn_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_layers,
            )
        model = get_peft_model(model, peft_config)
        if args.use_armt_pretrain:
            for n, p in model.named_parameters():
                if args.use_armt_pretrain:
                    if (
                        'memory' in n or 'sink' in n
                        or 'W_mq' in n or 'W_mk' in n or 'W_mv' in n or 'W_mb' in n
                    ):
                        p.requires_grad = True
                else:
                    if (
                        'memory_cell.memory' in n
                        or 'W_mq' in n or 'W_mk' in n or 'W_mv' in n or 'W_mb' in n
                    ):
                        p.requires_grad = True
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')
    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None and not args.use_armt_pretrain:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        mem_cell_args = dict(
            base_model=model,
            num_mem_tokens=args.num_mem_tokens,
        )
        # additional parameters for ARMT model
        if args.d_mem is not None:
            mem_cell_args['d_mem'] = args.d_mem
            mem_cell_args['wrap_pos'] = args.wrap_pos
            mem_cell_args['correction'] = not(args.no_correction)
            mem_cell_args['use_lora'] = args.use_lora
        if args.layers_attr is not None:
            mem_cell_args['layers_attr'] = args.layers_attr
        if args.attend_to_previous_input:
            mem_cell_args['attend_to_previous_input'] = args.attend_to_previous_input
        if args.n_heads is not None:
            mem_cell_args['n_heads'] = args.n_heads
        if args.use_sink:
            mem_cell_args['use_sink'] = args.use_sink
        if args.freeze_mem:
            mem_cell_args['freeze_mem'] = args.freeze_mem
        if args.compress_mem != 0:
            mem_cell_args['compress_mem'] = args.compress_mem
        if args.identity_frequency != 0:
            mem_cell_args['identity_frequency'] = args.identity_frequency
        if args.non_identity_layers is not None:
            mem_cell_args['non_identity_layers'] = args.non_identity_layers
        cell = memory_cell_cls(**mem_cell_args)
        if args.n_heads is not None:
            # case of model with sliding window
            model = recurrent_wrapper_cls(cell, 
                                          segment_size=block_size,
                                          max_n_segments=args.max_n_segments, 
                                          vary_n_segments=args.vary_n_segments,
                                          k2=args.k2,
                                          sliding_window=args.prev_seg_kv,
                                          return_all_logits=False,
                                          shift_mask_left=args.shift_mask_left,
            )
        else:
            model = recurrent_wrapper_cls(cell, 
                                          segment_size=block_size,
                                          max_n_segments=args.max_n_segments, 
                                          vary_n_segments=args.vary_n_segments,
                                          k2=args.k2,
                                          attend_to_previous_input=args.attend_to_previous_input,
                                          return_all_logits=False,
                                          shift_mask_left=args.shift_mask_left,
            )
        if args.distillator_cls is not None:
            print(f"Initializing distillator with alpha={args.alpha_distil}...")
            distillator = get_cls_by_name(args.distillator_cls)
            teacher_cls = get_cls_by_name(args.teacher_cls)
            teacher = teacher_cls.from_pretrained(args.pretrained_teacher)
            model = distillator(teacher, model, alpha_distil=args.alpha_distil)
        ## load cpt of rmt
        if args.model_cpt:
            if "safetensors" in args.model_cpt:
                print(model)
                from safetensors.torch import load_model
                load_model(model, args.model_cpt, device="cuda:0")
            else:
                if ".bin" in args.model_cpt:
                    model_cpt = args.model_cpt
                elif "model_best" in os.listdir(args.model_cpt):
                    model_cpt = os.path.join(args.model_cpt, "model_best", "pytorch_model.bin")
                else:
                    dir_files = os.listdir(args.model_cpt)
                    checkpoint_dir = [el for el in dir_files if "checkpoint-" in el][0]
                    model_cpt = os.path.join(args.model_cpt, checkpoint_dir, "pytorch_model.bin")
                cpt = torch.load(model_cpt, map_location='cpu')
                if "Inner" in model_cpt:
                    print(type(cpt))
                    #print(cpt)
                    new_dict = {}
                    pop_keys = []
                    for key in cpt.keys():
                        if "model.model.layers" in key:
                            if args.non_identity_layers is None:
                                new_key = key[12:]
                                new_dict[new_key] = cpt[key]
                            else:
                                buf_layers = args.non_identity_layers.split(";")
                                print(buf_layers)
                                if "W_mq" in key or "W_mk" in key or "W_mv" in key or "W_mb" in key:
                                    print(key, any([f".{layer}." in key for layer in buf_layers]), [f".{layer}." in key for layer in buf_layers])
                                    if any([f".{layer}." in key for layer in buf_layers]):
                                        new_key = key[12:]
                                        new_dict[new_key] = cpt[key]
                                    else:
                                        pop_keys.append(key)
                                    #else:
                                    #    new_key = key[12:]
                                    #    new_dict[new_key] = model.memory_cell[key]
                                else:
                                    new_key = key[12:]
                                    new_dict[new_key] = cpt[key]
                            #print(new_dict.keys())
                    for pop_key in pop_keys:
                        cpt.pop(pop_key)
                    cpt.update(new_dict)
                    model.memory_cell.load_state_dict(cpt, strict=True)
                else:
                    model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
        logger.info(f'Trainable parameters after adding RMT/ARMT:')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.use_armt_pretrain and args.model_cpt:
        if "safetensors" in args.model_cpt:
            print(model)
            from safetensors.torch import load_model
            load_model(model, args.model_cpt, device="cuda:0")
        else:
            if ".bin" in args.model_cpt:
                model_cpt = args.model_cpt
            elif "model_best" in os.listdir(args.model_cpt):
                model_cpt = os.path.join(args.model_cpt, "model_best", "pytorch_model.bin")
            else:
                dir_files = os.listdir(args.model_cpt)
                checkpoint_dir = [el for el in dir_files if "checkpoint-" in el][0]
                model_cpt = os.path.join(args.model_cpt, checkpoint_dir, "pytorch_model.bin")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
        logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
        logger.info(f'Trainable parameters after adding RMT/ARMT:')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.add_lora_to_armt:
        if args.lora_layers == "all-linear":
            exclude_armt = {"W_mq", "W_mk", "W_mv", "W_mb", "lm_head"}
            try:
                from transformers.pytorch_utils import Conv1D
            except Exception:
                Conv1D = ()
            lora_layers = sorted({
                name.split(".")[-1]
                for name, module in model.named_modules()
                if isinstance(module, (torch.nn.Linear, Conv1D))
                and name.split(".")[-1] not in exclude_armt
            })
        else:
            lora_layers = args.lora_layers.split(";") if ";" in args.lora_layers else args.lora_layers
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_attn_dim,
            lora_alpha=args.lora_attn_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_layers,
            )
        # add LoRA only to the inner model
        model.memory_cell.model = get_peft_model(model.memory_cell.model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.memory_cell.model.print_trainable_parameters()
        for n, p in model.named_parameters():
            if (
                'memory_cell.memory' in n
                or 'W_mq' in n or 'W_mk' in n or 'W_mv' in n or 'W_mb' in n
            ):
                p.requires_grad = True
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        #model.print_trainable_parameters()
        model.memory_cell.model.print_trainable_parameters()
        print(model)
        if args.model_cpt_v2:
            if "safetensors" in args.model_cpt_v2:
                print(model)
                from safetensors.torch import load_model
                load_model(model, args.model_cpt_v2, device="cuda:0")
            else:
                if ".bin" in args.model_cpt_v2:
                    model_cpt_v2 = args.model_cpt_v2
                elif "model_best" in os.listdir(args.model_cpt_v2):
                    model_cpt_v2 = os.path.join(args.model_cpt_v2, "model_best", "pytorch_model.bin")
                else:
                    dir_files = os.listdir(args.model_cpt_v2)
                    checkpoint_dir = [el for el in dir_files if "checkpoint-" in el][0]
                    model_cpt_v2 = os.path.join(args.model_cpt_v2, checkpoint_dir, "pytorch_model.bin")
                cpt = torch.load(model_cpt_v2, map_location='cpu')
                model.load_state_dict(cpt, strict=True)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt_v2}')
        logger.info(f'Trainable parameters after adding RMT/ARMT:')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.add_full_lora_to_armt:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_attn_dim,
            lora_alpha=args.lora_attn_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "W_mq", "W_mk", "W_mv", "W_mb"]
            )
        # add LoRA only to the inner model
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if 'memory' not in n and 'lora' not in n and 'adapter' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.tune_only_memory:
        for n, p in model.named_parameters():
            if 'memory_cell.memory' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.tune_only_armt:
        for n, p in model.named_parameters():
            if 'memory_cell.memory' not in n and 'W_mq' not in n and 'W_mk' not in n and 'W_mv' not in n and 'W_mb' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.no_tune_armt:
        for n, p in model.named_parameters():
            if 'memory_cell.memory' in n or 'W_mq' in n or 'W_mk' in n or 'W_mv' in n or 'W_mb' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    # fix the not-contiguous error
    def make_contiguous(module):
        with torch.no_grad():
            for param in module.parameters():
                param.set_(param.contiguous())
    make_contiguous(model)
    # now switch to HF trainer
    training_args_dict = {key: value for key, value in vars(args).items() if hasattr(SFTConfig('.'), key)}

    training_args_dict['remove_unused_columns'] = False
    training_args_dict['save_safetensors'] = False
    training_args_dict['bf16'] = True
    training_args_dict['label_names'] = ['labels']
    training_args_dict['eval_strategy'] = 'steps'
    if training_args_dict.get('per_device_train_batch_size') == 1:
        training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size')
    else:
        if args.task_name == "babilong":
            training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_eval_batch_size')
        else:
            training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size') // 2
    training_args_dict['eval_accumulation_steps'] = training_args_dict.get('gradient_accumulation_steps', 32)
    if args.d_mem is None:
        # for now, gradient checkpointing doesn't supported for ARMT
        training_args_dict['gradient_checkpointing'] = True
        training_args_dict['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
    training_args_dict['log_level'] = 'debug'
    training_args_dict['load_best_model_at_end'] = args.early_stopping_patience != -1
    training_args_dict['report_to'] = None
    #if args.task_name == "babilong":
    #    training_args_dict['dataset_kwargs'] = {"skip_prepare_dataset": True}
    if args.num_mem_tokens is not None:
        # fix max_seq_length warning
        training_args_dict["max_seq_length"] = args.sample_size
        model.to(torch.bfloat16)
    training_args = SFTConfig(**training_args_dict)
    
    print(model)
    
    #print("Model registered params")
    #for n, p in model.named_parameters():
    #    print(n, p.shape)
    print(training_args)
    #print(smth)


    collator = None
    if args.no_packing and "babilong" not in args.task_name:
        collator = collate_fn
    if "babilong" in args.task_name:
        if "babilong+" in args.task_name:
            collator = partial(collate_fn_mixin,
                               collate_func_bl=collate_fn_babilong_wrapped,
                               collate_func_other=collate_fn,
                               id_pad_value=id_pad_value,
                               padding_side=padding_side)
        else:
            collator = collate_fn_babilong_wrapped
            
    from pynvml import *

    def print_gpu_utilization():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # Adjust index if you have multiple GPUs
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")
    print_gpu_utilization()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    if args.use_lora and args.use_armt_pretrain:
        class SaveFullModelCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                # Only save on main process to avoid race conditions in distributed training
                if not state.is_world_process_zero:
                    return control
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                # Skip if checkpoint dir doesn't exist (trainer skipped save due to worse eval loss)
                if not os.path.isdir(checkpoint_dir):
                    return control
                model_to_save = kwargs.get("model")
                if model_to_save is not None:
                    if isinstance(model_to_save, PeftModel):
                        base_model = model_to_save.get_base_model()
                        logger.info(f"base_model.__class__: {base_model.__class__}")
                        
                        base_copy = copy.deepcopy(base_model)
                        peft_config = model_to_save.peft_config["default"] if isinstance(model_to_save.peft_config, dict) else model_to_save.peft_config
                        temp_peft = get_peft_model(base_copy, peft_config)
                        peft_state = get_peft_model_state_dict(model_to_save)
                        set_peft_model_state_dict(temp_peft, peft_state)
                        merged_model = temp_peft.merge_and_unload()
                        torch.save(merged_model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
                        if hasattr(merged_model, "config") and merged_model.config is not None:
                            merged_model.config.save_pretrained(checkpoint_dir)

                    else:
                        torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
                        if hasattr(model_to_save, "config") and model_to_save.config is not None:
                            model_to_save.config.save_pretrained(checkpoint_dir)
                return control
        trainer.add_callback(SaveFullModelCallback())
    logger.info(f"Trainer Gradient Checkpointing Enabled: {trainer.args.gradient_checkpointing}")
    if args.early_stopping_patience != -1:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        trainer.add_callback(early_stopping)
    if args.validate_only:
        print(train_dataset)
        print(train_dataset[0])
        print(train_dataset[0].keys())
        print(train_dataset[0]["attention_mask"].sum())
        print(np.mean([el["attention_mask"].sum().item() for el in train_dataset]))
        start_metrics = trainer.evaluate()
        logger.info(f"Metrics of initial model: {start_metrics}")
        with open(f"{args.output_dir}/val_scores.json", "w") as f:
            json.dump(start_metrics, f)
    #if not(training_args_dict["eval_on_start"]):
    #    start_metrics = trainer.evaluate()
    #    logger.info(f"Metrics of initial model: {start_metrics}")
    if not args.validate_only:
        trainer.train(resume_from_checkpoint=args.checkpoint)
