#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../../../
NP=1
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="2"
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling_sw_llama:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling_sw_llama:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
#NOISE_DATASET=EleutherAI/wikitext_document_level
#NOISE_DATASET_SPLIT=wikitext-103-raw-v1
NOISE_DATASET=pg19
METRIC=exact_match
TASK_NAMES=("babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k" "babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k" "babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k" "babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_8k")

# TASK_NAME=babilong+mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k

MODEL_NAME="unsloth/Llama-3.2-1B-Instruct"  # backbone model

# TODO: add model_cpt loading for all ckpt except first one
# TODO: maybe add EM evaluation (not so important)
ITERS=10000
TBS=64

D_MEM=64
N_HEADS=1
LORA_R=64
LORA_A=128
LORA_DROPOUT=0.1
WARMUP=0.0

for TASK_DATASET in "qa1_single-supporting-fact;qa2_two-supporting-facts;qa3_three-supporting-facts;qa4_two-arg-relations;qa5_three-arg-relations"
do

for LR in 1e-04
do

for SEGMENT_SIZE in 1024
do # size of one segment in tokens

MAX_N_SEGMENTSS=(2 3 5 8)
BSS=(4 2 1 1)
EVAL_BSS=(4 2 1 1)




#MAX_N_SEGMENTSS=( 6 8 16 32)
#BSS=( 8 8 4 2)

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BS=${BSS[j]}
EVAL_BS=${EVAL_BSS[j]}
# TODO: add task for various indices
TASK_NAME=${TASK_NAMES[j]}

for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
NUM_WORKERS=$((NP*1))

SCHEDULER=linear

for N in 1
do

K2=-1   # BPTT unroll length

GRADIENT_ACC_STEP=$(($TBS/($BS*$NP)))


# find prev checkpoint if needed


PREV_TASK_NAME=${TASK_NAMES[j]}
PREV_MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
PREV_INPUT_SEQ_LEN=$((SEGMENT_SIZE*PREV_MAX_N_SEGMENTS))
prev_model_path=$(find ../runs/finetune/bl_mix_mt_v2/armt_sw/${NOISE_DATASET}/${PREV_TASK_NAME}/${MODEL_NAME}/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_${LORA_R}_${LORA_A}_sink_SEGM_${PREV_MAX_N_SEGMENTS}x${SEGMENT_SIZE}_${PREV_INPUT_SEQ_LEN}_${D_MEM}_${LR}_${ITERS} -type f -regex '.*/pytorch_model.bin' | awk '{num=$0; sub(/.*checkpoint-/,"",num); sub(/[^0-9].*/,"",num); print num "\t" $0}' | sort -n -k1,1 | cut -f2- | head -1)

echo $prev_model_path
MODEL_CPT=$prev_model_path


echo $MODEL_CPT

echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRADIENT_ACC_STEP
#--noise_dataset_split $NOISE_DATASET_SPLIT \

accelerate launch --num_processes $NP --main_process_port 29506 --config_file ./accelerate_bf16.yaml run_finetuning_lm_rmt_hf_trainer.py \
        --task_name $TASK_NAME \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path ../babilong/data/tasks_1-20_v1-2/en-10k \
        --output_dir ../runs/finetune/eval_mt_v2_only_bl_mix_mt_v2/armt_sw/${NOISE_DATASET}/${TASK_NAME}/$MODEL_NAME/v8_assist_mask_no_sw_4gpu_1024_q15_all_lin_${LORA_R}_${LORA_A}_sink_SEGM_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_${SAMPLE_SIZE}_${D_MEM}_${LR}_${ITERS} \
        --model_cpt $MODEL_CPT \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --vary_n_segments \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size $EVAL_BS \
        --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 2 \
        --k1 -1 --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate  ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
        --data_n_workers $NUM_WORKERS \
        --logging_steps 100 --eval_steps 100 --save_steps 100 \
        --show_valid_examples 0 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --max_grad_norm 1.0 \
        --d_mem $D_MEM \
        --layers_attr model.layers \
        --mask_non_completion \
        --no_packing \
        --use_lora \
        --lora_layers "all-linear" \
        --lora_attn_dim ${LORA_R} \
        --lora_attn_alpha ${LORA_A} \
        --lora_dropout ${LORA_DROPOUT} \
        --warmup_ratio ${WARMUP} \
        --add_chat_template \
        --padding_side "left" \
        --mask_type "assistant" \
        --validate_only \
        --mixing_dataset_ratio 0.0 \
        --subsample_train 0.001
done
done
done
done
done
done
echo "done"
#        --use_lora \

#        --lora_layers "all-linear" \


