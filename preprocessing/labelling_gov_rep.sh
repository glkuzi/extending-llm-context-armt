export CUDA_VISIBLE_DEVICES="2"
# test labeling for 3-5 additional models to increase data diversity
# google/gemma-2-9b-it
# tiiuae/Falcon3-7B-Instruct
# Qwen/Qwen2.5-7B-Instruct
# microsoft/Phi-3-small-8k-instruct

#python label_long_data.py --use_vllm --seed 42 --start_index 45000 --num_samples 15000 --segment_size 4096 --input_dataset ../../data/long_context_raw/short_pajamaV2_1m_2k_1k --save_steps 100 --save_path ../../data/long_context_raw/llama31_8b_short_pajamaV2_1m_1k_2k_100k_labelled_div_4

#python label_long_data.py --use_vllm --seed 42 --start_index 900000 --num_samples 200000 --segment_size 4096 --input_dataset ../../data/long_context_raw/short_pajamaV2_1m_2k_1k --save_steps 1000 --save_path ../../data/long_context_raw/llama31_8b_short_pajamaV2_1m_1k_2k_100k_labelled_div_10

# +130k shift
#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_1


"""
python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen2.5-7B-Instruct' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_7b_1

python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_32b_1

python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_25_72b_1



python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'google/gemma-2-9b-it' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_gemma_2_9b_1
"""
"""
python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'microsoft/Phi-3.5-mini-instruct' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_phi_3_small_1


python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen3-4B-Instruct-2507' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_4b_1
"""
# now run more and bigger models - TODO check if these models work on our GPUs
# OOM
#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen3-30B-A3B-Instruct-2507' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_30b_a3b_1

#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'microsoft/phi-4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_phi_4_14b_1

# old transformers
#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'openai/gpt-oss-20b' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_gpt_oss_20b_1
# old transformers
#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'openai/gpt-oss-120b' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_gpt_oss_120b_1
# old transformers?
#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_30b_a3b_fp8_1

python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'mistralai/Mistral-7B-Instruct-v0.3' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_7b_v3_1

python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen3-30B-A3B-GPTQ-Int4' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_30b_a3b_gptq_int4_1

python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'Qwen/Qwen3-8B' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_qwen_3_8b_1

#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'mistralai/Ministral-3-14B-Instruct-2512' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_3_14b_1

#python label_gov_report.py --use_vllm --seed 42 --start_index 0 --num_samples 10000 --model_name 'mistralai/Ministral-3-8B-Instruct-2512' --segment_size 4096 --input_dataset ../../data/gov_report/gr_merged_with_gold_with_ids_debug --save_steps 100 --save_path ../../data/finetune_generated/gov_rep_base_gen_mistral_3_8b_1
cd ../scripts/finetune/gov_report
bash run_various_2s_gemma_bl_pars.sh