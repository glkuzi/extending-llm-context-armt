#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
# test labeling for 3-5 additional models to increase data diversity
#set -e
#ls;
##cd ../../
#eval "$(conda shell.bash hook)"
#conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
#cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/babilong_scripts/


#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
#set -e



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/two_gpu_v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x5120_5120_64_1e-05_10000/checkpoint-2200'
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
    --segment_size 5120 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/left_shift_v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_4x1024_4096_64_1e-04_10000/checkpoint-6000/pytorch_model.bin'
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


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/two_gpu_v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x4096_4096_64_1e-05_10000/checkpoint-5200'
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
    --segment_size 4096 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1

    
"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/left_shift_v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_4x1024_4096_64_1e-04_10000/checkpoint-4400/pytorch_model.bin'
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



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/two_gpu_v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x4096_4096_64_1e-05_10000/checkpoint-5200'
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
    --segment_size 2048 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1
"""
"""
# eval gemma w/o ft, and compare with ARMT
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='irodkin/ARMT-SmolLM2-360m-8x1024'
MODEL_CPT=''
TASKS=("qa1" "qa2")
#TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k")
#LENGTHS=("32k" "64k")
USE_CHAT_TEMPLATE=false
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
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


"""


"""
# eval gemma w/o ft, and compare with ARMT
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='irodkin/ARMT-Gemma-3-1B-IT-8x1024'
MODEL_CPT=''
TASKS=("qa4" "qa5")
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
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



# eval gemma w/o ft, and compare with ARMT
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT=''
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
#TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k" "64k")
#LENGTHS=("32k" "64k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --n_heads 1




# eval gemma w/o ft, and compare with ARMT
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='irodkin/ARMT-Gemma-3-1B-IT-8x1024'
MODEL_CPT=''
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
#TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k" "64k")
#LENGTHS=("32k" "64k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""

# eval gemma w/o ft, and compare with ARMT
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT=''
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
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --layers_attr model.layers \
    --n_heads 1





# eval gemma w/o ft, and compare with ARMT
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='irodkin/ARMT-Gemma-3-1B-IT-8x1024'
MODEL_CPT=''
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
    --mem_size 32 \
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""



















"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-8800/pytorch_model.bin'
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




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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





RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_128_256_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-8800/pytorch_model.bin'
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
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_128_256_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


"""

"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/unsloth/Llama-3.2-1B-Instruct/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-9300'
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
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
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




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
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





RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_128_256_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/unsloth/Llama-3.2-1B-Instruct/ckpt_fix_long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_128_256_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
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
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""



"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000//checkpoint-7000/pytorch_model.bin'
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



# gemma base with eval fix
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x2048_2048_64_1e-05_10000/checkpoint-7200'
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
    --segment_size 2048 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x1024_1024_64_1e-05_10000/checkpoint-7200'
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
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1
"""
"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-3900/pytorch_model.bin'
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


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-7200/pytorch_model.bin'
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
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/left_shift_v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-4600/pytorch_model.bin'
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

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/unsloth/Llama-3.2-1B-Instruct/long_val_v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-1400/pytorch_model.bin'
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
# eval gemma with eos fix
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/left_shift_v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-4600/pytorch_model.bin'
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
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-4800/pytorch_model.bin'
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



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
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

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-7400/pytorch_model.bin'
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



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/bl_mix_mt_v2/armt_sw/pg19/babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
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
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/left_shift_v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-3600/pytorch_model.bin'
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
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-6800/pytorch_model.bin'
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
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink
"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-3900/pytorch_model.bin'
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



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-7200/pytorch_model.bin'
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






RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-9000/pytorch_model.bin'
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
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-6800/pytorch_model.bin'
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
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_sw_1gpu_1024_q15_all_lin_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
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
    --n_heads 1 \
    --prev_seg_kv \
    --use_sink
"""
"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x2048_2048_64_1e-05_10000/checkpoint-7200'
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
    --segment_size 2048 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x1024_1024_64_1e-05_10000/checkpoint-7200'
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
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1


"""
"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x1024_1024_64_1e-05_10000/checkpoint-4400'
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
    --segment_size 1024 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_no_lora_sink_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-4000'
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
    --segment_size 8192 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1
"""



"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x8192_8192_64_1e-05_10000/checkpoint-4000'
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
    --segment_size 8192 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_armt_assist_mask_no_sw_1gpu_8192_q15_all_lin_64_128_sink_SEGM_1x2048_2048_64_1e-05_10000/checkpoint-4500'
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
    --segment_size 2048 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1
"""


"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_64_128_sink_SEGM_12x1024_12288_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_64_128_sink_SEGM_16x1024_16384_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1





RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_14_15_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-4600/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "14;15"



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_14_15_64_128_sink_SEGM_12x1024_12288_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "14;15"


    
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_14_15_64_128_sink_SEGM_16x1024_16384_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "14;15"



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_0_10_14_15_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-6100/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_0_10_14_15_64_128_sink_SEGM_12x1024_12288_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"







RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_assoc_0_10_14_15_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-6800/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_assoc_0_10_14_15_64_128_sink_SEGM_12x1024_12288_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"
"""

"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-3900/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_0_10_14_15_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-9500/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_0_10_14_15_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"

"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_assoc_0_10_14_15_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-5000/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_assoc_0_10_14_15_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"





RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_14_15_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "14;15"
"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_14_15_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-4900/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "14;15"
"""


"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-6000/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-4400/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_q_up_down_assoc_0_10_14_15_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-9500/pytorch_model.bin'
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
    --lora_layers "q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_0_10_14_15_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-9200/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "0;10;14;15"



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_all_lin_assoc_14_15_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-8500/pytorch_model.bin'
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
    --n_heads 1 \
    --non_identity_layers "14;15"
"""
"""
# ablate armt layers

for layer_ablate in "0;1;2;3" "4;5;6;7" "8;9;10;11" "12;13;14;15"
do
    RESULTS_FOLDER="../babilong_evals_bl_ft"
    DATASET_NAME="RMT-team/babilong"
    MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
    MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
        --armt_ablate "$layer_ablate" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1
done


for layer_ablate in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    RESULTS_FOLDER="../babilong_evals_bl_ft"
    DATASET_NAME="RMT-team/babilong"
    MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
    MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
        --armt_ablate "$layer_ablate" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1
done

"""
"""
# sequentially ablate lora

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "k_proj;v_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "k_proj;v_proj;o_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "k_proj;v_proj;o_proj;gate_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "k_proj;v_proj;o_proj;gate_proj;q_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "k_proj;v_proj;o_proj;gate_proj;q_proj;up_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "k_proj;v_proj;o_proj;gate_proj;q_proj;up_proj;down_proj" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


"""









"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "mlp" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --lora_ablate "self_attn" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


for layer_ablate in q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
do
    RESULTS_FOLDER="../babilong_evals_bl_ft"
    DATASET_NAME="RMT-team/babilong"
    MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
    MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
        --lora_ablate "$layer_ablate" \
        --d_mem 64 \
        --layers_attr model.layers \
        --n_heads 1
done
"""
"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-7900/pytorch_model.bin'
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


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-6100/pytorch_model.bin'
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



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-5500/pytorch_model.bin'
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



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_64_128_sink_SEGM_12x1024_12288_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_12x512_6144_test_32_64_1e-04_10000/checkpoint-2500/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_base_64_128_sink_SEGM_12x1024_12288_64_1e-04_10000/checkpoint-4400/pytorch_model.bin'
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
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""


"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_12x512_6144_test_32_64_1e-04_10000/checkpoint-1500/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_base_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-9300/pytorch_model.bin'
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
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""


"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_base_64_128_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-9200/pytorch_model.bin'
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
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_base_64_128_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-5300/pytorch_model.bin'
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
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_base_64_128_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-2600/pytorch_model.bin'
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
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_1024_q15_base_64_128_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_no_sw_1gpu_sw_q15_all_lin_64_128_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-6900/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_no_sw_1gpu_sw_q15_all_lin_64_128_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_no_sw_1gpu_sw_q15_all_lin_64_128_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-6200/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_no_sw_1gpu_sw_q15_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
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
    --segment_size 512 \
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


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_no_lora_sink_SEGM_16x512_8192_64_1e-04_10000/checkpoint-6900/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_no_lora_sink_SEGM_12x512_6144_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_no_lora_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_base_16_32_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-8400/pytorch_model.bin'
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
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_no_lora_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-9800/pytorch_model.bin'
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
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_no_lora_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-7700/pytorch_model.bin'
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
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_no_lora_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
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
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""



"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_base_16_32_sink_SEGM_5x1024_5120_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_base_16_32_sink_SEGM_8x1024_8192_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1





RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_base_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-8000/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_base_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-5500/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_base_16_32_sink_SEGM_3x1024_3072_64_1e-04_10000/checkpoint-8000/pytorch_model.bin'
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
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-2600/pytorch_model.bin'
TASKS=("qa1")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
TASKS=("qa1")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1






RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_shift_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_1024_q15_base_16_32_sink_SEGM_2x1024_2048_64_1e-04_10000/checkpoint-8400/pytorch_model.bin'
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
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
# eval qa1-5, big lora, assist mask




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-4200/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
#TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("32k" "64k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-8000/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
#TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("32k" "64k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_16x512_8192_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_12x512_6144_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-7200/pytorch_model.bin'
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
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
    --segment_size 512 \
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
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-4200/pytorch_model.bin'
#TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_4gpu_sw_q15_all_lin_64_128_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-8000/pytorch_model.bin'
#TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
TASKS=("qa6" "qa7" "qa8" "qa9" "qa10")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
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


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_12x512_6144_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



"""
"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-200/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-2600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""



"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-4600/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1





RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1




RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-2300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


"""
"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_targ_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-2900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-1800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""


"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-700/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""



"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_assist_mask_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1


RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v8_no_sw_1gpu_sw_q1_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-1200/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='openai-community/gpt2'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/openai-community/gpt2/gpt2_v3_no_lora_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-2300/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
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
    --legacy_bl_prompt \
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr transformer.h \
    --n_heads 1



# eval gpt-2 with legacy prompt

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='openai-community/gpt2'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/openai-community/gpt2/gpt2_v3_no_lora_all_lin_64_128_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-3100/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
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
    --legacy_bl_prompt \
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr transformer.h \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='openai-community/gpt2'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/openai-community/gpt2/gpt2_v3_no_lora_all_lin_64_128_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-2800/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
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
    --legacy_bl_prompt \
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr transformer.h \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='openai-community/gpt2'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/openai-community/gpt2/gpt2_v3_no_lora_all_lin_64_128_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-3900/pytorch_model.bin'
TASKS=("qa1")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
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
    --legacy_bl_prompt \
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr transformer.h \
    --n_heads 1



"""

"""
#v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-1100
# res after pad fix
# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint

RESULTS_FOLDER="../DEBUG_babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-1900/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



# same, but w/o tags in prompt

RESULTS_FOLDER="../DEBUG_babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_2x512_1024_64_1e-04_10000/checkpoint-1900/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1



RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v7_no_sw_wm_1gpu_sw_q15_all_lin_16_32_sink_SEGM_5x512_2560_64_1e-04_10000/checkpoint-500/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""

"""
# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../DEBUG_babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/google/gemma-3-1b-it/v5_wm_no_sw_q15_all_lin_16_32_sink_SEGM_0x512_0_64_1e-04_10000/checkpoint-1000/pytorch_model.bin'
TASKS=("qa1" )
LENGTHS=("0k" )
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 64 \
    --use_peft 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_layers "all-linear" \
    --d_mem 64 \
    --layers_attr model.layers \
    --n_heads 1
"""
"""
# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../DEBUG_babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v6_wm_1gpu_sw_q15_all_lin_64_128_sink_SEGM_3x512_1536_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
TASKS=("qa1" )
LENGTHS=("0k" )
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
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
# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v6_wm_1gpu_sw_q15_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-2200/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
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




# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v6_wm_1gpu_sw_q15_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-04_10000/checkpoint-2200/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
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
# 0-shot, legacy prompt - same as in train - to reproduce, set legacy_prompt=True in code
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v3_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-05_10000/checkpoint-8500/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
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

RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=false
USE_DOUBLE_QUEST=false

API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_DOUBLE_QUEST" == true ] && echo "--use_double_question" ) \
    --api_url "$API_URL"



# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v3_all_lin_64_128_sink_SEGM_16x512_8192_64_1e-05_10000/checkpoint-7900/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
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




# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v3_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-05_10000/checkpoint-8500/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
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




# eval BL for 16x512 ARMT-SW
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals_bl_ft"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/finetune/babilong_multitask/armt_sw/pg19/babilong/unsloth/Llama-3.2-1B-Instruct/v3_all_lin_64_128_sink_SEGM_8x512_4096_64_1e-05_10000/checkpoint-8500/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
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
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_2k_4k_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1800/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers




# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_4k_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_4x1024_4032_64_1e-05_10000/checkpoint-2400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers
"""

"""
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers


# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers





# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/from_scratch_model_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers


"""
"""

# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/from_scratch_model_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers






# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/from_scratch_model_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers





"""






"""
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-2400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers






# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-2400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers




# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-2400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers


"""

"""


# TODO: When run, set separate save file
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x992_1952_64_1e-05_10000/checkpoint-1400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 992 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --max_new_tokens 100



"""








"""


# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x992_1952_64_1e-05_10000/checkpoint-1400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 992 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers






# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x992_1952_64_1e-05_10000/checkpoint-1400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 992 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers



# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x992_1952_64_1e-05_10000/checkpoint-1400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 992 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers

"""



"""
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_2k/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_2000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers






# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_2k/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_2000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers




# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_2k/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_2000/checkpoint-600/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers

"""


"""

# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora_ext/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_100000/checkpoint-20000/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1008 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers







# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora/armt_1k_2s_llama_base/synth_reduced_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1700/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1008 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers

"""
"""

# eval ARMT from IT llama-8b on smoltalk, up to 2s with mask fix, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/armt_ift/smol_comp_multi_mask_no_pack/smoltalk_long_05_mix/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64/checkpoint-2000/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers
"""

"""
# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-9200/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers
"""

"""
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME="../../runs/loss_abaltion_lora/llama_2048_base/synth_reduced_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_1x2048_2048_1e-05_10000/checkpoint-8100"
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=false
USE_DOUBLE_QUEST=false

API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_DOUBLE_QUEST" == true ] && echo "--use_double_question" ) \
    --api_url "$API_URL"
"""
"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_sliding_llama3.2-1B_ctx512x4_step24000_pg19.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
USE_QUEST_FIRST=false
USE_ANSWER_FILL=true
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
    --api_url "$API_URL" \
    --mem_size 32 \
    --segment_size 512 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 4 \
    --layers_attr model.layers \
    --attend_to_previous_input 1




# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_sliding_llama3.2-1B_ctx512x4_step24000_pg19.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
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
    --api_url "$API_URL" \
    --mem_size 32 \
    --segment_size 512 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 4 \
    --layers_attr model.layers \
    --attend_to_previous_input 1



RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
TASKS=("qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
USE_QUEST_FIRST=false
USE_DOUBLE_QUEST=false

API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_DOUBLE_QUEST" == true ] && echo "--use_double_question" ) \
    --api_url "$API_URL"

"""

"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora/armt_1k_2s_llama_base/synth_reduced_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1700/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers



RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
USE_DOUBLE_QUEST=false

API_URL=""

echo running $MODEL_NAME on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    $( [ "$USE_QUEST_FIRST" == true ] && echo "--use_quest_first_template" ) \
    $( [ "$USE_DOUBLE_QUEST" == true ] && echo "--use_double_question" ) \
    --api_url "$API_URL"




# eval ARMT from base llama on synth loss with q1-only, 1m, prelim checkpoint
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-5000/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers

"""


"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-4800/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
USE_QUEST_FIRST=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers
"""
