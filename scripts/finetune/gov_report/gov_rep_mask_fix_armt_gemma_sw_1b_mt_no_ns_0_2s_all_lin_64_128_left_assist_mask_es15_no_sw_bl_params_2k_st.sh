#!/usr/bin/env bash
set -e
cd ../../../
NP=1
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="0"
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling_sw_llama:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling_sw_llama:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=gov_report/gov_report_0k_2k

# Now trying with packing and w/o flash attn, cause training doesn't work otherwise
ITERS=2000
TBS=64
INPUT_SIZE=1024

MAX_N_SEGMENTSS=(2)
MEMORY_SIZE=16
D_MEM=64
N_HEADS=1
BSS=(2)
LORA_R=64
LORA_A=128
LORA_DROPOUT=0.1
WARMUP=0.0
for N in 2
do

for MODEL_NAME in "google/gemma-3-1b-it"
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
# for ARMT we have memory embs only at the end of the sequence
INPUT_SEQ_LEN=$(((INPUT_SIZE-MEMORY_SIZE)*MAX_N_SEGMENTS))
K2=-1
K2_PREV=-1
BS=${BSS[j]}

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in 1e-04
do

GRADIENT_ACC_STEP=$(($TBS/($BS*$NP)))
echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM
accelerate launch --num_processes $NP --config_file ./accelerate_bf16.yaml run_finetuning_lm_rmt_hf_trainer.py \
        --task_name $TASK_NAME \
        --output_dir ../runs/finetune/armt_sw/${TASK_NAME}/$MODEL_NAME/steps_2k_bl_pars_mask_fix_assist_mask_es15_all_lin_${LORA_R}_${LORA_A}_sink_SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_${D_MEM}_${LR}_${ITERS} \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --sample_size $INPUT_SEQ_LEN \
        --segment_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --per_device_train_batch_size $BS --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 2 \
        --k1 -1 --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate  ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --logging_steps 20 --eval_steps 20 --save_steps 20 \
        --show_valid_examples 0 \
        --early_stopping_patience 30 \
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
        --padding_side "left" \
        --mask_type "assistant"
done
done
done
done
done
done
echo "done"
