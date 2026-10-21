#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e

# eval mt-v2 with top-4 only
# eval gr from 4s
# eval gr with 20%
# eval mt-v2-script all cl 

# now eval ft on pre-train w/o cl with pre-selected + new yarn
# new yarn after adapt

# new yarn
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k//home/jovyan/gkuzmin/rmt_it/runs/finetune/base_model/pg19_subset_yarn_full/HuggingFaceTB/SmolLM2-360M-Instruct/full_pg19_yarn_hf_4_sub_train_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x32768_32768_64_1e-04_400/checkpoint-400/from_full_pg19_adapt_yarn_hf_4_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-300'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --use_yarn_hf




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus/gov_report_0k_8k//home/jovyan/gkuzmin/rmt_it/runs/finetune/base_model/pg19_subset_yarn_full/HuggingFaceTB/SmolLM2-360M-Instruct/full_pg19_yarn_hf_4_sub_train_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x32768_32768_64_1e-04_400/checkpoint-400/from_full_pg19_adapt_yarn_hf_4_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-8500'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --use_yarn_hf




"""
# new yarn
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/yarn_hf_4_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-1100'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --use_yarn_hf




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/yarn_hf_4_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-7800'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --use_yarn_hf



# without cl
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/no_cl_fixed_from_new_pretrain_top_N_universal_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_10000/checkpoint-7900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"

"""


"""

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/no_cl_fixed_from_new_pretrain_top_N_universal_old_impl_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_5000/checkpoint-4000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"


# with cl
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/fixed_from_new_pretrain_top_N_universal_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-4900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"


"""







"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/home/jovyan/gkuzmin/rmt_it/runs/finetune/base_model/pg19_subset_yarn_full/HuggingFaceTB/SmolLM2-360M-Instruct/full_pg19_yarn_16_sub_train_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x32768_32768_64_1e-04_400/checkpoint-400/from_full_pg19_adapt_yarn_16_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-300'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --use_yarn



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/fixed_from_new_pretrain_top_N_universal_old_impl_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_5000/checkpoint-2750/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/fixed_from_new_pretrain_top_N_universal_old_impl_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_3e-05_5000/checkpoint-1800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/fixed_from_new_pretrain_top_N_universal_old_impl_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-05_5000/checkpoint-4000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"




RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/google/gemma-3-1b-it/fixed_from_new_pretrain_top_N_universal_old_impl_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_10000/checkpoint-7000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"




RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/google/gemma-3-1b-it/fixed_from_new_pretrain_top_N_universal_old_impl_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"

"""

"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus/gov_report_0k_8k//home/jovyan/gkuzmin/rmt_it/runs/finetune/base_model/pg19_subset_yarn_full/HuggingFaceTB/SmolLM2-360M-Instruct/full_pg19_yarn_16_sub_train_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x32768_32768_64_1e-04_400/checkpoint-400/from_full_pg19_adapt_yarn_16_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-7800'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --use_yarn
"""
"""
# eval RMT
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/rmt_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_None_1e-05_5000/checkpoint-4650/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear"
"""

"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/rmt_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_None_3e-05_5000/checkpoint-3750/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear"
"""

"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/rmt_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_None_3e-05_5000/checkpoint-3200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear"


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/rmt_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_None_1e-04_5000/checkpoint-4050/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear"
"""

"""

RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/mem_32_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-4700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-3000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-2500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-2200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;15;16;18;19;20;21;22;25;26;27;28;29;30;31"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-4700/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done



RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_no_cl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_10000/checkpoint-800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;24;25;26;27;28;29;30;31"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-4700/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


# top-1 - 0-ht only
for layer_ablate in "1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3650/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done
"""
"""


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""


"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/HuggingFaceTB/SmolLM2-360M-Instruct/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-6200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

# eval top-n pre-selected smollm 2s and 4s
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-7100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "8;16;17;23;24;31"


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-04_10000/checkpoint-4400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "8;16;17;23;24;31"


# eval pretrain 2s and 4s

RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/google/gemma-3-1b-it/from_new_pretrain_old_impl_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_10000/checkpoint-7100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/google/gemma-3-1b-it/from_new_pretrain_old_impl_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
# now smollm2 for mt-v2 , 8s
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-4700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-3900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "8;16;17;23;24;31"
"""


"""

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-05_5000/checkpoint-1850/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/from_new_pretrain_no_cl_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_5000/checkpoint-4000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/from_new_pretrain_no_cl_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_5000/checkpoint-2450/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
# top-4 - w/o 0, 23, 30, 20
for layer_ablate in "1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;21;22;24;25;26;27;28;29;31"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3650/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done



# top-5 - w/o 0, 23, 30, 20, 27
for layer_ablate in "1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;21;22;24;25;26;28;29;31"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3650/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-05_5000/checkpoint-700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""



"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/from_new_pretrain_old_impl_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_3e-05_5000/checkpoint-1800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/google/gemma-3-1b-it/bl_pars_mem_32_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_10000/checkpoint-7100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/google/gemma-3-1b-it/bl_pars_mem_32_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/from_new_pretrain_old_impl_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_5000/checkpoint-2700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/from_new_pretrain_no_cl_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_5000/checkpoint-1000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/yarn_16_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-7800'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --use_yarn




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/from_new_pretrain_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-2500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --n_heads 1 \
    --use_armt_pretrain "irodkin/InnerLoopARMTForCausalLM_run_35"
"""

"""

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "8;16;17;23;24;31"


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-3600'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/top_N_assoc_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-5100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"



RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-04_10000/checkpoint-4400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
# eval 1b and 8b 0-shot with and w/o instructions
# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus_plus/gov_report_0k_8k/google/gemma-3-1b-it/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_20000/checkpoint-7000'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers
"""
"""

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-3950/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "8;16;17;23;24;31"



RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/top_N_assoc_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-4200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"
"""

"""


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-04_10000/checkpoint-4200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""

"""

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-3400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-4250/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "8;16;17;23;24;31"



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-3150/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "8;16;17;23;24;31"



RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/top_N_assoc_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-1500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/HuggingFaceTB/SmolLM2-360M-Instruct/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-6200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



for layer_ablate in {14..31}
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3650/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


"""

















"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3650/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



for layer_ablate in "0;1;2;3;4;5;6;7" "8;9;10;11;12;13;14;15" "16;17;18;19;20;21;22;23" "24;25;26;27;28;29;30;31"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3650/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


for layer_ablate in {0..31}
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3650/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done
"""
"""


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



for layer_ablate in "0;1;2;3;4;5;6;7" "8;9;10;11;12;13;14;15" "16;17;18;19;20;21;22;23" "24;25;26;27;28;29;30;31"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-2000/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done

"""

"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-1150/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/recalc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-3900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/top_N_assoc_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-4100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/top_N_assoc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-1900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/top_N_assoc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-1850/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus_plus/gov_report_0k_2k/google/gemma-3-1b-it/steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-5000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus_plus/gov_report_0k_4k/google/gemma-3-1b-it/steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_10000/checkpoint-6300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/google/gemma-3-1b-it/top_N_assoc_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-7100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/google/gemma-3-1b-it/top_N_assoc_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-04_10000/checkpoint-4500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "7;19;13;14;25"

"""

"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus/gov_report_0k_8k/HuggingFaceTB/SmolLM2-360M-Instruct/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-7800'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/from_4s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus_subset_20_perc/gov_report_0k_8k/google/gemma-3-1b-it/perc_20_top_4_assoc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-1550/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "9;13;14;25"





# eval gemma with eos fix
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/inst_pretrain_armt_sw/armt_1k_2s_llama_base/synth_reduced_qa_inst_llama8b_mix_pajama_all_mix_1k_1920/google/gemma-3-1b-it/mask_fix_assist_mask_no_sw_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-6500/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
#TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k" "64k")
#LENGTHS=("32k" "64k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=false
USE_ANSWER_FILL=false
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    --no_tags_prompt \
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


# eval gemma with eos fix
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/inst_pretrain_armt_sw/armt_1k_2s_llama_base/synth_reduced_qa_inst_llama8b_mix_2k_3840_mix_2s_v4_140k_025_2s/google/gemma-3-1b-it/assist_mask_no_sw_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-3400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
#TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k" "64k")
#LENGTHS=("32k" "64k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=false
USE_ANSWER_FILL=false
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    --no_tags_prompt \
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""
"""


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-1850/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/HuggingFaceTB/SmolLM2-360M-Instruct/steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-4100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



"""
"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/top_4_assoc_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-4200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "14;18;19;25"


"""
"""
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/top_4_assoc_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-2400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "14;18;19;25"
"""


"""

RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/top_4_assoc_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-1200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "14;18;19;25"

RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/google/gemma-3-1b-it/top_4_assoc_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-04_10000/checkpoint-4400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "14;18;19;25"


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/google/gemma-3-1b-it/top_4_assoc_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-7000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "14;18;19;25"












RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2_script_version/manytypes_flat_stacked_no_empty_sv_0k_2k_val_2k/google/gemma-3-1b-it/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-9900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2_script_version/manytypes_flat_stacked_no_empty_sv_0k_4k_val_4k/google/gemma-3-1b-it/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_10000/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2_script_version/manytypes_flat_stacked_no_empty_sv_0k_8k_val_8k/google/gemma-3-1b-it/recalc_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-4400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""
echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus_subset_20_perc/gov_report_0k_4k/google/gemma-3-1b-it/perc_20_top_4_assoc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-1100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "9;13;14;25"


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus_subset_20_perc/gov_report_0k_2k/google/gemma-3-1b-it/perc_20_top_4_assoc_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "9;13;14;25"


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/from_4s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-1700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/from_4s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-2700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""
"""
# eval models trained woth ablated layers only


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/top_4_assoc_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "9;13;14;25"

"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/top_4_assoc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-3750/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "9;13;14;25"



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/top_4_assoc_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-4200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "9;13;14;25"

"""


"""

# best 8s armt on GR-100+
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1





for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
do
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-2000/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""
    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done

"""

"""


# also top-5 - 14, 25, 9, 13, 18
for layer_ablate in "0;1;2;3;4;5;6;7;8;10;11;12;15;16;17;19;20;21;22;23;24"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3500/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done

"""
"""
# by halfs
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12" "13;14;15;16;17;18;19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-4700/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""
    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


# 14th only
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;15;16;17;18;19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-4700/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""
    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done
"""

"""
# base test, top-3 by importance - fixed layers for bal_sup
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;15;16;17;19;20;21;22;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-5100/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""
    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done

# also top-3
for layer_ablate in "0;1;2;3;4;5;6;7;8;10;11;12;13;15;16;17;18;19;20;21;22;23;24"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3500/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done
"""

"""
# base test, only 14 layer
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;15;16;17;18;19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-5100/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""
    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done




for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;15;16;17;18;19;20;21;22;23;24;25"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3500/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done



# base test, top-4 by importance
for layer_ablate in "0;1;2;3;4;5;6;7;9;10;11;12;15;16;17;19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-5100/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""
    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"
    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done




for layer_ablate in "0;1;2;3;4;5;6;7;8;10;11;12;15;16;17;18;19;20;21;22;23;24"
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3500/pytorch_model.bin'
    TASKS=("qa1")
    LENGTHS=("0k")
    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=false
    USE_EXAMPLES=false
    USE_POST_PROMPT=false
    USE_QUEST_FIRST=true
    USE_ANSWER_FILL=false
    USE_CUAD_TEMPLATE=true
    API_URL=""

    echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

    python run_rmt_model_on_qa.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_cpt "$MODEL_CPT" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
        $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
        $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
        --api_url "$API_URL" \
        --max_new_tokens 256 \
        --mem_size 16 \
        --segment_size 1024 \
        --max_n_segments 64 \
        --use_peft 1 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.1 \
        --lora_layers "all-linear" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done
"""

"""
# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k_val_16k/google/gemma-3-1b-it/from_2s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


"""
"""
# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/from_2s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-2700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/from_2s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-3750/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/from_2s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-1850/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_4k/unsloth/gemma-3-1b-it/from_2s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_4000/checkpoint-1240/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_8k/unsloth/gemma-3-1b-it/from_2s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_4000/checkpoint-1360/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

# eval 1b and 8b 0-shot with and w/o instructions
# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-4000'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers
"""



"""
# prelim res for base model
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-3500'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_5000/checkpoint-3500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-3750/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_5000/checkpoint-4100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_2k/google/gemma-3-1b-it/from_2s_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_4000/checkpoint-1880/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report_100_gen/gov_report_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-05_8000/checkpoint-7940'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_4000/checkpoint-1320/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_4k/google/gemma-3-1b-it/steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_4000/checkpoint-1240/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_2k/google/gemma-3-1b-it/steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_4000/checkpoint-1880/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

# now rev cl

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_2k/google/gemma-3-1b-it/cl_rev_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x512_1984_64_3e-05_2000/checkpoint-620/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 128 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_1k/google/gemma-3-1b-it/cl_rev_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x256_960_64_5e-05_2000/checkpoint-180/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 256 \
    --max_n_segments 256 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_512/google/gemma-3-1b-it/cl_rev_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x256_480_64_1e-04_2000/checkpoint-60/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 128 \
    --max_n_segments 512 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report/gov_report_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-05_4000/checkpoint-1540'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/google/gemma-3-1b-it/steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_2000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k_val_32k/google/gemma-3-1b-it/long_val_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_2000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_2k/google/gemma-3-1b-it/steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_2000/checkpoint-300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/google/gemma-3-1b-it/steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_2000/checkpoint-540/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""

"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/google/gemma-3-1b-it/lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_2k/google/gemma-3-1b-it/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""

# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/google/gemma-3-1b-it/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_10000/checkpoint-800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
# for left pad llama3 1b
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/left_pad_assist_mask_es15_no_sw_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-1200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




# for left pad llama3 1b
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/left_pad_assist_mask_es15_no_sw_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""

"""
# for left pad llama3 1b
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k_val_32k/unsloth/Llama-3.2-1B-Instruct/left_pad_assist_mask_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink
"""


"""
# eval gov-rep armt only
# eval gov-rep-sub50 - base all-lin lora, armt-sw, full ft

# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_50_gen/gov_report_0k_8k/google/gemma-3-1b-it/all_lin_64_128_no_sw_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-3300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_50_gen/gov_report_0k_4k/google/gemma-3-1b-it/all_lin_64_128_no_sw_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-4500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""

# eval gov-rep llama with assistant mask


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/left_pad_assist_mask_all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-2300'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers


# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/left_pad_assist_mask_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# for left pad llama3 1b
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/left_pad_assist_mask_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink

"""

"""
# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/left_pad_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink



# now eval gemma3-1b


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report/gov_report_0k_8k/google/gemma-3-1b-it/all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-2300'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers



# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report/gov_report_0k_8k/google/gemma-3-1b-it/full_ft_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-1800/pytorch_model.bin'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers


# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/google/gemma-3-1b-it/all_lin_64_128_no_sw_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-1200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/google/gemma-3-1b-it/all_lin_64_128_no_sw_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1







# gemma for gov_rep 50


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report_50_gen/gov_report_0k_8k/google/gemma-3-1b-it/all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-8300'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers



# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/gov_report_50_gen/gov_report_0k_8k/google/gemma-3-1b-it/full_ft_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-6500/pytorch_model.bin'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers

"""
"""

# for left pad llama3 1b
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/left_pad_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink
"""

"""
# eval 2s GR models - to find out, how to train on 2s

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_2k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_id4_sink_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-2300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink \
    --identity_frequency 4





# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_2k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_no_sw_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1







# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_50_gen/gov_report_0k_2k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-6600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_2k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-7200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink


"""
"""

# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_2k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_2k/unsloth/Llama-3.2-1B-Instruct/from_2s_inst_st2s_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink


"""


"""
# eval gov-rep armt only
# eval gov-rep-sub50 - base all-lin lora, armt-sw, full ft

# from all-lin 50 gen
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_no_sw_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-1200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_no_sw_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




# from all-lin 50 gen
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_50_gen/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/es_15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_50_gen/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/es_15_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-3000/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink
"""


"""
# eval gov-rep-sub50 - base all-lin lora, armt-sw, full ft

# from all-lin 50 gen
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_50_gen/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128 50 gen
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_50_gen/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-2900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink

"""

"""


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report_50_gen/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-8300'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers



# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report_50_gen/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/full_ft_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-8300/pytorch_model.bin'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers
"""
"""
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/from_4s_inst_st4s_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-1200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink
"""


"""
# from 2s st4s
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/from_2s_inst_st4s_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/from_2s_inst_st4s_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink





# from 4s st2s
# 8s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/from_4s_inst_st2s_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-1200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/from_4s_inst_st2s_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink

"""

"""
# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/from_4s_inst_st4s_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink


"""







"""

# it model, tuned with full attention
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT=''
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers





RESULTS_FOLDER="../qa_evals_gov_rep_64k/inst"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT=''
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return the original question and your answer without any additional explanation or generation."




RESULTS_FOLDER="../qa_evals_gov_rep_64k/inst"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT=''
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return the original question and your answer without any additional explanation or generation."


# it model, tuned with full attention
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT=''
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers




# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-1800'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers



# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_SEGM_1x4096_4096_64_1e-05_10000/checkpoint-1600'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 4096 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers





# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink




# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_comp_256_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-2100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink \
    --compress_mem 256



# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_id4_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-2700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink \
    --identity_frequency 4


# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_id4_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink \
    --identity_frequency 4



# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/base_model/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/full_ft_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-1800/pytorch_model.bin'
TASKS=("qa1")
# "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")
# "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --segment_size 8192 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers



# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_comp_256_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-1800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink \
    --compress_mem 256



# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink



# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_4k/unsloth/Llama-3.2-1B-Instruct/from_2s_inst_st2s_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink



    
# 1b exps - all lin 64 128
# 4s 1024
RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report/gov_report_0k_8k/unsloth/Llama-3.2-1B-Instruct/from_2s_inst_st2s_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_ANSWER_FILL=false
USE_CUAD_TEMPLATE=true
API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_rmt_model_on_qa.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_cpt "$MODEL_CPT" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_ANSWER_FILL" == true ] && echo "--use_answer_fill" ) \
    $( [ "$USE_CUAD_TEMPLATE" == true ] && echo "--use_cuad_template" ) \
    --api_url "$API_URL" \
    --max_new_tokens 256 \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink



"""




















