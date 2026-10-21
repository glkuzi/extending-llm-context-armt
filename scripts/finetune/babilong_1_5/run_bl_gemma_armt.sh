#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
# test labeling for 3-5 additional models to increase data diversity
set -e
ls;
nvidia-smi;
#cd ../../
eval "$(conda shell.bash hook)"
conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/scripts/finetune/babilong_1_5/

#bash armt_gemma3_1b_bl_all_lin_64_128_cl_2_16_bs_new_gen_4_gpu_q15_v8_gpt2like_pad_fix_mask_assistant_1024_eos_fix_rerun.sh;
bash armt_gemma3_1b_bl_all_lin_64_128_cl_2_16_bs_new_gen_4_gpu_q15_v8_gpt2like_pad_fix_mask_assistant_1024_eos_fix_4_only.sh;