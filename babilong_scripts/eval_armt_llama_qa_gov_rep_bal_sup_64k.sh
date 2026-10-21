#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
# test labeling for 3-5 additional models to increase data diversity
set -e
ls;
#cd ../../
eval "$(conda shell.bash hook)"
conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/babilong_scripts/








# eval models with ft only and mem 32
for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
do
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/mem_32_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-05_5000/checkpoint-1850/pytorch_model.bin'
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
        --armt_ablate "$layer_ablate"
done


for layer_ablate in {0..25}
do
    RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
    DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
    MODEL_NAME='google/gemma-3-1b-it'
    MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/mem_32_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-05_5000/checkpoint-1850/pytorch_model.bin'
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
        --armt_ablate "$layer_ablate"
done













"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_8k/google/gemma-3-1b-it/mem_32_steps_2k_lower_lr_from_low_lr_4s_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-05_5000/checkpoint-1850/pytorch_model.bin'
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
    --n_heads 1



RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k/google/gemma-3-1b-it/mem_32_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_10000/checkpoint-6500/pytorch_model.bin'
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
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k/google/gemma-3-1b-it/mem_32_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_1e-04_10000/checkpoint-400/pytorch_model.bin'
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
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_2k/google/gemma-3-1b-it/mem_32_steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_2x1024_1984_64_1e-04_5000/checkpoint-4100/pytorch_model.bin'
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
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/google/gemma-3-1b-it/mem_32_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_3968_64_3e-05_5000/checkpoint-1900/pytorch_model.bin'
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
MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k/google/gemma-3-1b-it/from_new_pretrain_old_impl_no_cl_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_7936_64_1e-04_10000/checkpoint-5700/pytorch_model.bin'
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
    --n_heads 1



for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
do
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
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


for layer_ablate in {0..25}
do
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
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done
"""

"""
# now run for mt-v2 on smollm




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




for layer_ablate in "0;1;2;3;4;5;6;7" "8;9;10;11;12;13;14;15" "16;17;18;19;20;21;22;23" "24;25;26;27;28;29;30;31"
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




# SET FINAL CKPT
for layer_ablate in {0..31}
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



"""
"""
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return your answer without any additional explanation or generation."


# it model, tuned with full attention
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return your answer without any additional explanation or generation."



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers


# it model, tuned with full attention
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers

"""


"""
# SET FINAL CKPT
for layer_ablate in {0..31}
do
    # best 8s armt on GR-100+
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
    MODEL_NAME='HuggingFaceTB/SmolLM2-360M-Instruct'
    MODEL_CPT='../../runs/finetune/armt_sw/mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k_val_16k/HuggingFaceTB/SmolLM2-360M-Instruct/lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_8x1024_8064_64_3e-05_10000/checkpoint-3600/pytorch_model.bin'
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
MODEL_CPT='../../runs/finetune/armt_sw/gov_report_100_gen_plus/gov_report_0k_4k/HuggingFaceTB/SmolLM2-360M-Instruct/top_N_assoc_steps_2k_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin_64_128_sink_SEGM_4x1024_4032_64_3e-05_5000/checkpoint-2900/pytorch_model.bin'
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

"""
for layer_ablate in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
    # best 8s armt mt_v2
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
# now same for bal support

# top-4 and top-5 - 14, 18, 5/19/25
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;15;16;17;20;21;22;23;24"
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


# top-4 and top-5 - 14, 18, 5/19/25 - here is top-5, add remoced 5
for layer_ablate in "0;1;2;3;4;6;7;8;9;10;11;12;13;15;16;17;20;21;22;23;24"
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
for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
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




for layer_ablate in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
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
# by halfs
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12" "13;14;15;16;17;18;19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
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




for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12" "13;14;15;16;17;18;19;20;21;22;23;24;25"
do
    # best 8s armt on GR-100+
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
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done




# now same for bal support

for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
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


for layer_ablate in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
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




for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
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


for layer_ablate in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
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




# by halfs
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12" "13;14;15;16;17;18;19;20;21;22;23;24;25"
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




for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12" "13;14;15;16;17;18;19;20;21;22;23;24;25"
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




# base test, only 14 layer
for layer_ablate in "0;1;2;3;4;5;6;7;8;9;10;11;12;13;15;16;17;18;19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
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
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done



# base test, top-4 by importance
for layer_ablate in "0;1;2;3;4;5;6;7;9;10;11;12;15;16;17;19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
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
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done

"""
"""

for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
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


for layer_ablate in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
    # best 8s armt mt_v2
    RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
    DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
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




for layer_ablate in "0;1;2;3;4;5;6" "7;8;9;10;11;12" "13;14;15;16;17;18" "19;20;21;22;23;24;25"
do
    # best 8s armt on GR-100+
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
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


for layer_ablate in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
    # best 8s armt on GR-100+
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
        --n_heads 1 \
        --armt_ablate "$layer_ablate"
done


"""

"""
# it model, tuned with full attention - all tasks, both support versions

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return your answer without any additional explanation or generation."


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return your answer without any additional explanation or generation."


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return your answer without any additional explanation or generation."


# it model, tuned with full attention
RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2_bal_sup"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_v2_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers \
    --pre_question_prompt "Given the following context and question, answer the question. Return your answer without any additional explanation or generation."

"""


"""
# it model, tuned with full attention

RESULTS_FOLDER="../qa_evals_gov_rep_64k"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only"
MODEL_NAME='google/gemma-3-1b-it'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
MODEL_NAME='google/gemma-3-1b-it'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers


RESULTS_FOLDER="../qa_evals_mt_0k_64k_600_v2"
DATASET_NAME="../../data/mt_prepared_v2/test_mt_flat_stacked_no_empty_sv_0k_64k_600_v2"
MODEL_NAME='google/gemma-3-1b-it'
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
    --segment_size 8196 \
    --max_n_segments 128 \
    --use_peft 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --layers_attr model.layers
"""
"""
# from 2s
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
    --n_heads 1


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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


# it model, tuned with all-lin 64 128
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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



RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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

bash eval_armt_llama_qa_mt_64k_bal_sup_64k.sh;
"""

"""

# now rev cl

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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


RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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

RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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





RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup/inst"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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




RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup/inst"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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
RESULTS_FOLDER="../qa_evals_gov_rep_64k_bal_sup"
DATASET_NAME="../../data/gov_report/gov_report_0k_64k_test_only_bal_support"
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




















