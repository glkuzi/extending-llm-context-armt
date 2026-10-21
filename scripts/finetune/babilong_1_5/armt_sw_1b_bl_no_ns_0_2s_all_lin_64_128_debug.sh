#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../../../
NP=1
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0"
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling_sw_llama:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling_sw_llama:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
NOISE_DATASET=EleutherAI/wikitext_document_level
NOISE_DATASET_SPLIT=wikitext-103-raw-v1
METRIC=exact_match
TASK_NAME=babilong

MODEL_NAME="unsloth/Llama-3.2-1B-Instruct"  # backbone model

# TODO: add model_cpt loading for all ckpt except first one
# TODO: maybe add EM evaluation (not so important)
ITERS=10000
TBS=64

D_MEM=64
N_HEADS=1
LORA_R=64
LORA_A=128

for TASK_DATASET in "qa1_single-supporting-fact;qa2_two-supporting-facts;qa3_three-supporting-facts;qa4_two-arg-relations;qa5_three-arg-relations"
do

for LR in 1e-05
do

for SEGMENT_SIZE in 512
do # size of one segment in tokens

MAX_N_SEGMENTSS=(16 2 4)
BSS=(1 4 4)
EVAL_BSS=(1 2 2)

#MAX_N_SEGMENTSS=( 6 8 16 32)
#BSS=( 8 8 4 2)

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BS=${BSS[j]}
EVAL_BS=${EVAL_BSS[j]}

j1=$((j-1))
SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}

j2=$((j-2))
SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}


for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))

SCHEDULER=linear

for N in 1
do

K2=-1   # BPTT unroll length

GRADIENT_ACC_STEP=$(($TBS/($BS*$NP)))
echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS


accelerate launch --num_processes $NP --main_process_port 29501 --config_file ./accelerate_bf16.yaml run_finetuning_lm_rmt_hf_trainer.py \
        --task_name $TASK_NAME \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --noise_dataset_split $NOISE_DATASET_SPLIT \
        --babi_path ../babilong/data/tasks_1-20_v1-2/en-10k \
        --output_dir ../runs/finetune/DEBUG_babilong_multitask/armt_sw/${TASK_NAME}/$MODEL_NAME/all_lin_${LORA_R}_${LORA_A}_sink_SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_${D_MEM}_${LR}_${ITERS} \
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
        --data_n_workers 4 \
        --logging_steps 10 --eval_steps 10 --save_steps 10 \
        --show_valid_examples 0 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --max_grad_norm 1.0 \
        --tokenizer_for_chat_template "meta-llama/Llama-3.1-8B-Instruct" \
        --d_mem $D_MEM \
        --layers_attr model.layers \
        --mask_non_completion \
        --no_packing \
        --n_heads $N_HEADS \
        --prev_seg_kv \
        --use_sink \
        --use_lora \
        --lora_layers "all-linear" \
        --lora_attn_dim ${LORA_R} \
        --lora_attn_alpha ${LORA_A}
done
done
done
done
done
done
echo "done"


