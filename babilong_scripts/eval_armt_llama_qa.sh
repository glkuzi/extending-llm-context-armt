#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e

# Eval Llama3.2-1b base + IFT versions from various datasets
# use versions w/o chat template





# eval ARMT from base llama on synth loss with q1-only
# 8b model, trained on 1m synth, after on 1k-2k qa inst synth and CL on 2k-4k qa inst synth
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_2k_4k_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_4x1024_4032_64_1e-05_10000/checkpoint-1800/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100





# eval ARMT from base llama on synth loss with q1-only
# 8b model, trained on 1m synth, after on 1k-2k qa inst synth and CL on 1k-4k qa inst synth
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_4k_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_4x1024_4032_64_1e-05_10000/checkpoint-2400/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100




"""

# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/unsloth/Llama-3.2-1B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-3900/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100 \
    --pre_question_prompt 'Answer the following question based on the context. The answer should have following structure:\n 1. Restate the question to set context.\n 2. Provide a concise and complete final answer.'
"""






"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100 \
    --pre_question_prompt 'Answer the following question based on the context. The answer should have following structure:\n 1. Restate the question to set context.\n 2. Provide a concise and complete final answer.'









# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/from_scratch_model_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100 \
    --pre_question_prompt 'Answer the following question based on the context. The answer should have following structure:\n 1. Restate the question to set context.\n 2. Provide a concise and complete final answer.'




# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100

# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/from_scratch_model_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""

"""

# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1100/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100

"""

"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/meta-llama/Llama-3.1-8B-Instruct/from_scratch_model_SEGM_2x1024_2016_64_1e-05_10000/checkpoint-1600/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""

"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_qa_inst_1k_1920_big/unsloth/Llama-3.2-1B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-3900/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""





"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x992_1952_64_1e-05_10000/checkpoint-1400/pytorch_model.bin'
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
    --dataset_samples 100



# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-2400/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100


"""



"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_1792_big/unsloth/Llama-3.2-1B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-4800/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""

"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_2k/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_2000/checkpoint-600/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""
"""

# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_2k/unsloth/Llama-3.2-1B-Instruct/SEGM_2x1024_2016_64_1e-05_2000/checkpoint-600/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""

"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/loss_abaltion_cot/armt_1k_2s_llama_base/synth_reduced_cot_1k_2k/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64_1e-05_2000/checkpoint-710/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""

"""

# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora_ext/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_100000/checkpoint-29000/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100



# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora_ext/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_100000/checkpoint-20000/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100


# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/armt_ift/smol_comp_multi_mask_no_pack/smoltalk_long_05_mix/unsloth/Llama-3.2-1B-Instruct/SEGM_2x1024_2016_64/checkpoint-10000/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --attend_to_previous_input 1 \
    --dataset_samples 100
"""




"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_lora/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.1-8B-Instruct/SEGM_2x1024_2016_64_1e-05_10000/checkpoint-9200/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""

"""
# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_no_lora_prev_input/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/unsloth/Llama-3.2-1B-Instruct/SEGM_4x512_1984_64_1e-05_10000/checkpoint-9700/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 512 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --attend_to_previous_input 1 \
    --dataset_samples 100



# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='meta-llama/Llama-3.2-1B'
MODEL_CPT='../../runs/loss_abaltion_no_lora_ext/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/meta-llama/Llama-3.2-1B/SEGM_2x1024_2016_64_1e-05_100000/checkpoint-88000/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100




# eval ARMT from base llama on synth loss with q1-only
RESULTS_FOLDER="../qa_evals"
DATASET_NAME="../../data/synth_reduced_1k_1792_test_only"
MODEL_NAME='unsloth/Llama-3.2-1B-Instruct'
MODEL_CPT='../../runs/loss_abaltion_no_lora_ext/armt_1k_2s_llama_base/synth_reduced_1k_1792_1m_big/unsloth/Llama-3.2-1B-Instruct/SEGM_2x1024_2016_64_1e-05_100000/checkpoint-85000/pytorch_model.bin'
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
    --api_url "$API_URL" \
    --mem_size 16 \
    --segment_size 1024 \
    --max_n_segments 32 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --d_mem 64 \
    --layers_attr model.layers \
    --dataset_samples 100
"""

