#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e

# Eval Llama3.1-8b - on small (100 samples) BABILong
# NA - base with Q1
# IT with Q1 and CT +
# ARMT-80k with Q1 and CT +
# ARMT-1m with Q1 and CT -
# IT-80k with Q1 and CT +
# IT with all +


# compare with ultralong llama8b
# nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='nvidia/Llama-3.1-8B-UltraLong-1M-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")
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
# try to compare res with prolong-tuned llama

RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="RMT-team/babilong"
MODEL_NAME='princeton-nlp/Llama-3-8B-ProLong-512k-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")
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
