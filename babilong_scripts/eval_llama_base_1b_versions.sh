#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e

# Eval Llama3.2-1b base + IFT versions from various datasets
# use versions w/o chat template



# eval for ARMT 2s from LM-2s-19500 on Flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/TEST_LOSS_MASK_armt_all_s_ift_from_LM_hf_trainer/LM_pg19_19500_64_16/dolly/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64/checkpoint-300/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers


"""
# eval for ARMT 2s from LM-2s-19500 on Flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/TEST_LOSS_MASK_armt_all_s_ift_from_base_hf_trainer/dolly/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64/checkpoint-200/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers
"""


"""
# eval for ARMT 2s from LM-2s-19500 on Flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_LM_hf_trainer/LM_pg19_4s_19500_64_16/dolly/meta-llama/Llama-3.2-1B/SEGM_4x1024_4032_64/checkpoint-300/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers
"""


"""


# eval for ARMT 2s from LM-2s-19500 on Flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_LM_hf_trainer/LM_pg19_19500_64_16/flan/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64-inter/checkpoint-22000-inter/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers
"""

"""

# eval for ARMT 5s from CL on Flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_base_hf_trainer/flan/meta-llama/Llama-3.2-1B/SEGM_5x1024_5040_64/checkpoint-48000-inter/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers





# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_ctx4096_step19500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers





# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_ctx4096_step19500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers



# eval for ARMT 8s from LM-2s-19500 on Dolly-Mix-05-8s
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_LM_hf_trainer_from_scratch/LM_pg19_19500_64_16/dolly_long_05_mix/meta-llama/Llama-3.2-1B/SEGM_8x1024_8064_64/checkpoint-200/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers




# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_step49500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers





# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_step49500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers






# eval for flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_base_hf_trainer_from_scratch/dolly_long_05_mix/unsloth/Llama-3.2-1B-Instruct/SEGM_8x1024_8064_64/checkpoint-150/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers



# eval for flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_LM_hf_trainer/LM_pg19_49500_64_16/dolly/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64/checkpoint-400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers
"""




"""


# eval for flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_base_hf_trainer/flan/meta-llama/Llama-3.2-1B/SEGM_3x1024_3024_64/checkpoint-94000/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers




# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_step19500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers



# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_step31500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers





# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_step19500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers



# eval for LM-ARMT
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../data/pretrained_models/ARMT-llama3.2-1B/armt_llama3.2-1B_step31500.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers







# eval for flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_LM_hf_trainer/LM_pg19_19500_64_16/dolly/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64/checkpoint-400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers


# eval for flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_LM_hf_trainer/LM_pg19_31500_64_16/dolly/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64/checkpoint-400/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers

"""


"""

RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
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
    --api_url "$API_URL"




RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
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
    --api_url "$API_URL"









RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
TASKS=("qa3")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=true
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
    --api_url "$API_URL"
"""
"""

# eval for flan
RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/armt_ift/armt_all_s_ift_from_base_hf_trainer/flan/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64/checkpoint-100000/pytorch_model.bin'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
#LENGTHS=("32k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=true
USE_POST_PROMPT=true
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --d_mem 64 \
    --layers_attr model.layers




RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
TASKS=("qa1" "qa2" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=true
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
    --api_url "$API_URL"




# now eval instruct llama

RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=true
USE_INSTRUCTION=true
USE_EXAMPLES=false
USE_POST_PROMPT=true
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
    --api_url "$API_URL"



RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
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
    --api_url "$API_URL"


RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
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
    --api_url "$API_URL"



"""



"""

RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
TASKS=("qa1" "qa2" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=true
USE_EXAMPLES=false
USE_POST_PROMPT=true
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
    --api_url "$API_URL"



RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
TASKS=("qa3")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=true
USE_EXAMPLES=false
USE_POST_PROMPT=true
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
    --api_url "$API_URL"


RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
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
    --api_url "$API_URL"


RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=true
USE_POST_PROMPT=false
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
    --api_url "$API_URL"



RESULTS_FOLDER="../babilong_evals"
DATASET_NAME="booydar/babilong-1k-samples"
MODEL_NAME='meta-llama/Llama-3.2-1B'
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k")
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=true
USE_EXAMPLES=false
USE_POST_PROMPT=false
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
    --api_url "$API_URL"

"""

