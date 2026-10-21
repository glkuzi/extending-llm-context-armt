#!/usr/bin/env bash
set -e
cd ../../../
NP=1
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="0"

python eval_ce_fineweb.py \
        --model ../data/pretrained_models/InnerLoopARMTForCausalLM_run_35_pt/pytorch_model.bin \
        --load-legacy-armt "google/gemma-3-1b-it"