#!/usr/bin/env bash
set -e
cd ../../../
#ls

PREV_DATASET=mt_prepared/manytypes_flat_stacked_no_empty_sv_0k_2k
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
D_MEM=64
MEMORY_SIZE=16
INPUT_SIZE=1024
ITERS=10000
LR=1e-05
PREV_MAX_N_SEGMENTS=2
PREV_INPUT_SEQ_LEN=$(((INPUT_SIZE-MEMORY_SIZE)*PREV_MAX_N_SEGMENTS))
model_path=$(find ../runs/finetune/armt_sw/${PREV_DATASET}/${MODEL_NAME}/from_inst_sink_SEGM_${PREV_MAX_N_SEGMENTS}x${INPUT_SIZE}_${PREV_INPUT_SEQ_LEN}_${D_MEM}_${LR}_${ITERS} -type f -regex '.*/pytorch_model.bin' | sort -t '\0' -n | head -1)
model_path=$(find ../runs/finetune/armt_sw/${PREV_DATASET}/${MODEL_NAME}/sink_SEGM_2x1024_2016_64_1e-05_10000 -type f -regex '.*/pytorch_model.bin' | sort -t '\0' -n | head -1)

echo $model_path
echo