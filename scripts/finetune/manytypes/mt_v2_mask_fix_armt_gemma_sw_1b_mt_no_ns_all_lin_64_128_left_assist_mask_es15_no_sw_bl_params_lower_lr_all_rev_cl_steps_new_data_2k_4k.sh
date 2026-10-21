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

# Curriculum learning parameters
MAX_N_SEGMENTSS=(4 4 4)
INPUT_SIZES=(256 512 1024)
LRS=(1e-4 1e-4 3e-5)
BSS=(2 2 1)
TASK_NAMES=(mt_prepared_v2_script_version/manytypes_flat_stacked_no_empty_sv_0k_1k_val_1k mt_prepared_v2_script_version/manytypes_flat_stacked_no_empty_sv_0k_2k_val_2k mt_prepared_v2_script_version/manytypes_flat_stacked_no_empty_sv_0k_4k_val_4k)
#TASK_NAMES=(mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_256 mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_512 mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_1k mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_2k mt_prepared_v2/manytypes_flat_stacked_no_empty_sv_0k_4k)
OUTPUT_DIR_PATTERNS=(new_data_cl_rev_v2_bl_pars_mask_fix_assist_mask_es15_all_lin new_data_cl_rev_v2_bl_pars_mask_fix_assist_mask_es15_all_lin new_data_cl_rev_v2_lower_lr_bl_pars_mask_fix_assist_mask_es15_all_lin)
CONFIG_FILES=(./accelerate_bf16.yaml ./accelerate_bf16.yaml ./accelerate_bf16.yaml)
CUDA_VISIBLE_DEVICES_ARRAY=("0" "0" "0")
NP_ARRAY=(1 1 1)

ITERS_ARRAY=(10000 10000 10000)
EARLY_STOPPING_PATIENCE_ARRAY=(15 15 15)

# Common parameters
TBS=64
MEMORY_SIZE=16
D_MEM=64
N_HEADS=1
LORA_R=64
LORA_A=128
LORA_DROPOUT=0.1
WARMUP=0.0
MODEL_NAME="google/gemma-3-1b-it"

#for N in 12
for N in 2
do

for (( i=2; i<${#MAX_N_SEGMENTSS[@]}; i++ ))
do

MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[i]}
INPUT_SIZE=${INPUT_SIZES[i]}
LR=${LRS[i]}
BS=${BSS[i]}
TASK_NAME=${TASK_NAMES[i]}
OUTPUT_DIR_PATTERN=${OUTPUT_DIR_PATTERNS[i]}
CONFIG_FILE=${CONFIG_FILES[i]}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_ARRAY[i]}
NP=${NP_ARRAY[i]}
ITERS=${ITERS_ARRAY[i]}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE_ARRAY[i]}

# for ARMT we have memory embs only at the end of the sequence
INPUT_SEQ_LEN=$(((INPUT_SIZE-MEMORY_SIZE)*MAX_N_SEGMENTS))
K2=-1
K2_PREV=-1

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

# Determine model checkpoint path
MODEL_CPT=None
if [ $i -eq 0 ]; then
  # First stage: start from pretrained model
  MODEL_CPT=None
else
  # Subsequent stages: use checkpoint from previous stage
  PREV_MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[i-1]}
  PREV_LR=${LRS[i-1]}
  PREV_TASK_NAME=${TASK_NAMES[i-1]}
  PREV_OUTPUT_DIR_PATTERN=${OUTPUT_DIR_PATTERNS[i-1]}
  PREV_ITERS=${ITERS_ARRAY[i-1]}
  PREV_INPUT_SIZE=${INPUT_SIZES[i-1]}
  PREV_INPUT_SEQ_LEN=$(((PREV_INPUT_SIZE-MEMORY_SIZE)*PREV_MAX_N_SEGMENTS))
  
  # Find the latest checkpoint from previous stage
  prev_output_dir="../runs/finetune/armt_sw/${PREV_TASK_NAME}/${MODEL_NAME}/${PREV_OUTPUT_DIR_PATTERN}_${LORA_R}_${LORA_A}_sink_SEGM_${PREV_MAX_N_SEGMENTS}x${PREV_INPUT_SIZE}_${PREV_INPUT_SEQ_LEN}_${D_MEM}_${PREV_LR}_${PREV_ITERS}"
  echo "Previous output directory: $prev_output_dir"
  MODEL_CPT=$(find ${prev_output_dir} -type f -regex '.*/pytorch_model.bin' | awk '{num=$0; sub(/.*checkpoint-/,"",num); sub(/[^0-9].*/,"",num); print num "\t" $0}' | sort -n -k1,1 | cut -f2- | head -1)
  echo "Using checkpoint from previous stage: $MODEL_CPT"
fi

GRADIENT_ACC_STEP=$(($TBS/($BS*$NP)))
echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM

# Build accelerate command
ACCELERATE_CMD="accelerate launch --num_processes $NP --main_process_port 29507 --config_file $CONFIG_FILE run_finetuning_lm_rmt_hf_trainer.py"

# Build output directory
OUTPUT_DIR="../runs/finetune/armt_sw/${TASK_NAME}/${MODEL_NAME}/${OUTPUT_DIR_PATTERN}_${LORA_R}_${LORA_A}_sink_SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_${D_MEM}_${LR}_${ITERS}"

# Execute training
$ACCELERATE_CMD \
        --task_name $TASK_NAME \
        --output_dir $OUTPUT_DIR \
        $( [ "$MODEL_CPT" != "None" ] && echo "--model_cpt $MODEL_CPT" ) \
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
        --logging_steps 100 --eval_steps 100 --save_steps 100 \
        --show_valid_examples 0 \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
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

echo "Done"