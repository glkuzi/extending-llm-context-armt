#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
# test labeling for 3-5 additional models to increase data diversity
# google/gemma-2-9b-it
# tiiuae/Falcon3-7B-Instruct
# Qwen/Qwen2.5-7B-Instruct
# microsoft/Phi-3-small-8k-instruct
ls;
#cd ../../
eval "$(conda shell.bash hook)"
conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/preprocessing/

##python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 10000 --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_2
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 1000 --model_name 'Qwen/Qwen2.5-7B-Instruct' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_7b_2
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 2000 --model_name 'Qwen/Qwen2.5-7B-Instruct' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_7b_25
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_2
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_32b_2
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_72b_2
#
#
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'google/gemma-2-9b-it' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_gemma_2_9b_2
#
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'microsoft/Phi-3.5-mini-instruct' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_phi_3_small_2
#
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'Qwen/Qwen3-4B-Instruct-2507' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_4b_2
#
## now run more and bigger models - TODO check if these models work on our GPUs
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'microsoft/phi-4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_phi_4_14b_2
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'mistralai/Mistral-7B-Instruct-v0.3' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_7b_v3_2
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'Qwen/Qwen3-30B-A3B-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_30b_a3b_gptq_int4_2
#
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'Qwen/Qwen3-8B' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_8b_2
#

# new batch
# broken
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'mistralai/Ministral-3-14B-Instruct-2512' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_3_14b_2
#finished
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 3000 --model_name 'mistralai/Ministral-3-8B-Instruct-2512' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_3_8b_2


#finished
#python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'Qwen/Qwen2.5-7B-Instruct' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_7b_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_32b_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_72b_3



python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'google/gemma-2-9b-it' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_gemma_2_9b_3


python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'microsoft/Phi-3.5-mini-instruct' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_phi_3_small_3


python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'Qwen/Qwen3-4B-Instruct-2507' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_4b_3

# now run more and bigger models - TODO check if these models work on our GPUs

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'microsoft/phi-4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_phi_4_14b_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'mistralai/Mistral-7B-Instruct-v0.3' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_7b_v3_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'Qwen/Qwen3-30B-A3B-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_30b_a3b_gptq_int4_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'Qwen/Qwen3-8B' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_8b_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'mistralai/Ministral-3-14B-Instruct-2512' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_3_14b_3

python label_gov_report.py --use_vllm --seed 17 --start_index 0 --num_samples 7000 --model_name 'mistralai/Ministral-3-8B-Instruct-2512' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_3_8b_3
