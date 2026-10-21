#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"
# test labeling for 3-5 additional models to increase data diversity
#set -e
#ls;
#cd ../../
#eval "$(conda shell.bash hook)"
#conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
#cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/babilong_scripts/

# SmolLM ARMT evaluations (non-loop only)



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/HuggingFaceTB/SmolLM2-360M-Instruct/from_new_pretrain_old_impl_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_5000/checkpoint-4250/pytorch_model.bin'
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
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/from_new_pretrain_old_impl_recalc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_3e-05_5000/checkpoint-3850/pytorch_model.bin'
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
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/HuggingFaceTB/SmolLM2-360M-Instruct/from_new_pretrain_old_impl_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_10000/checkpoint-7100/pytorch_model.bin'
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
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/from_new_pretrain_old_impl_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_1e-04_10000/checkpoint-4400/pytorch_model.bin'
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


# best 8s armt on MT
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/from_new_pretrain_old_impl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_3e-05_10000/checkpoint-4700/pytorch_model.bin'
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
    --n_heads 1

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


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_1e-04_10000/checkpoint-2200/pytorch_model.bin'
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