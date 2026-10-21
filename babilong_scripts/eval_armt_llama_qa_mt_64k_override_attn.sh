#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e


# Run gemma models with flash_attn and eager to check the speed and performance differences
# Run base model, and 8s model with fixed masks


# with flash_attn
# 8s 1024, no sw
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-200/pytorch_model.bin'
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
    --override_attn "flash_attention_2"


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-10000'
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
    --override_attn "flash_attention_2"




# with eager
# 8s 1024, no sw
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-200/pytorch_model.bin'
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
    --override_attn "eager"


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-10000'
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
    --override_attn "eager"

"""


# with sdpa
# 8s 1024, no sw
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_1e-05_10000/checkpoint-200/pytorch_model.bin'
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
    --override_attn "sdpa"


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-10000'
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
    --override_attn "sdpa"

"""