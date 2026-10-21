#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
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
NOISE_DATASET=pg19
METRIC=exact_match

MODEL_NAME="unsloth/Llama-3.2-1B-Instruct"  # backbone model

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

MAX_N_SEGMENTSS=(0 1 2 4 6 8 16 32)
BSS=(32 32 16 16 8 8 4 2)

MAX_N_SEGMENTSS=( 6 8 16 32)
BSS=( 8 8 4 2)

for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BS=${BSS[j]}

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


echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

accelerate launch --config_file ./accelerate_bf16.yaml --main_process_port 29008 run_finetuning_babilong_rmt_multitask.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/jovyan/rmt/babilong/data/tasks_1-20_v1-2/en-10k \
        --model_path /home/jovyan/rmt/runs/babilong_multitask/qa1-10/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_N_SEGMENTS}-${MAX_N_SEGMENTS}/run_$N \
        --model_cpt /home/jovyan/rmt/runs/babilong_multitask/qa1-10/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}/run_$N/model_best \
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
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --save_best \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0
        
done
done
done
done
done
done
done
done
echo "done"




























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
TASK_NAME=gov_report/gov_report_0k_2k

# Now trying with packing and w/o flash attn, cause training doesn't work otherwise
ITERS=10000
TBS=32
INPUT_SIZE=1024

MAX_N_SEGMENTSS=(2)
MEMORY_SIZE=16
D_MEM=64
N_HEADS=1
BSS=(1)
LORA_R=64
LORA_A=128
COMPRESS_MEM=256
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
accelerate launch --num_processes $NP --main_process_port 29501 --config_file ./accelerate_bf16.yaml run_finetuning_lm_rmt_hf_trainer.py \
        --task_name $TASK_NAME \
        --output_dir ../runs/finetune/armt_sw/${TASK_NAME}/$MODEL_NAME/all_lin_${LORA_R}_${LORA_A}_comp_${COMPRESS_MEM}_sink_SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_${D_MEM}_${LR}_${ITERS} \
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
        --lora_layers "all-linear" \
        --group_by_length \
        --lora_attn_dim ${LORA_R} \
        --lora_attn_alpha ${LORA_A} \
        --compress_mem ${COMPRESS_MEM}
done
done
done
done
done
done
echo "done"
