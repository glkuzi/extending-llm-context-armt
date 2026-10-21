#!/usr/bin/env bash
set -e
cd ../../../
NP=4
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling_sw_llama:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling_sw_llama:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=mt_prepared/manytypes_flat_stacked_no_empty_sv_0k_2k

# Now trying with packing and w/o flash attn, cause training doesn't work otherwise
ITERS=10000
TBS=32
INPUT_SIZE=512

MAX_N_SEGMENTSS=(2)
MEMORY_SIZE=16
D_MEM=64
N_HEADS=1
BSS=(1)
for N in 2
do

for MODEL_NAME in "unsloth/Llama-3.2-1B-Instruct"
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
# for ARMT we have memory embs only at the end of the sequence
INPUT_SEQ_LEN=$(((INPUT_SIZE-MEMORY_SIZE)*MAX_N_SEGMENTS))
K2=2
K2_PREV=2
BS=${BSS[j]}

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in 1e-05
do

GRADIENT_ACC_STEP=$(($TBS/($BS*$NP)))
echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM
accelerate launch --num_processes $NP --config_file ./accelerate_bf16.yaml run_finetuning_lm_rmt_hf_trainer.py \
        --task_name $TASK_NAME \
        --output_dir ../runs/finetune/armt_sw/${TASK_NAME}/$MODEL_NAME/sink_SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_${D_MEM}_${LR}_${ITERS} \
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
        --optimizer AdamW  --weight_decay 0.001 \
        --learning_rate  ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --logging_steps 100 --eval_steps 100 --save_steps 100 \
        --show_valid_examples 0 \
        --early_stopping_patience 5 \
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
        --group_by_length
done
done
done
done
done
done
echo "done"
