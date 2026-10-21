#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"

##!/bin/bash
#export CUDA_VISIBLE_DEVICES="0"
## test labeling for 3-5 additional models to increase data diversity
#set -e
#ls;
##cd ../../
#eval "$(conda shell.bash hook)"
#conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
#cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/babilong_scripts/




RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
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
    --clear_memory \
    --non_identity_layers "7;19;13;14;25"



RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
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
    --clear_memory \
    --non_identity_layers "7;19;13;14;25"



RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
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
    --clear_memory \
    --non_identity_layers "7;19;13;14;25"



"""

# base model, on 128k
RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_no_cl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_10000/checkpoint-9700/pytorch_model.bin'
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
    --clear_memory




# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/base_model/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/google/gemma-3-1b-it/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_SEGM_1x8192_8192_64_1e-04_10000/checkpoint-9700'
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
    --clear_memory



# eval 2s from scratch and from 2s
# 2s 1024, no sw
RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/google/gemma-3-1b-it/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_2016_64_1e-04_10000/checkpoint-7000/pytorch_model.bin'
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
    --clear_memory





# eval 2s models, plus 4s from best 2s
RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/google/gemma-3-1b-it/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_10000/checkpoint-4000/pytorch_model.bin'
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
    --clear_memory



# eval 4s lower lr and 8s lower lr from high lr
RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
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
    --clear_memory






RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
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
    --n_heads 1 \
    --clear_memory


RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
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
    --n_heads 1 \
    --clear_memory










RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-5100/pytorch_model.bin'
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
    --clear_memory




RESULTS_FOLDER="../qa_evals_mt_0k_128k_1500_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_128k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_no_cl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_10000/checkpoint-9700/pytorch_model.bin'
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
    --clear_memory

"""