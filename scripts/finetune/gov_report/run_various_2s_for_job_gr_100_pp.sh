#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"
# test labeling for 3-5 additional models to increase data diversity
set -e
ls;
nvidia-smi;
#cd ../../
eval "$(conda shell.bash hook)"
conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/scripts/finetune/gov_report/


#bash gov_rep_100_plus_plus_mask_fix_armt_gemma_sw_1b_mt_no_ns_0_2s_all_lin_64_128_left_assist_mask_es15_no_sw_bl_params.sh;
#bash gov_rep_100_plus_plus_mask_fix_armt_gemma_sw_1b_mt_no_ns_0_4s_all_lin_64_128_left_assist_mask_es15_no_sw_bl_params_lower_lr.sh;
#bash gov_rep_100_plus_plus_mask_fix_armt_gemma_sw_1b_mt_no_ns_0_8s_all_lin_64_128_left_assist_mask_es15_no_sw_bl_params_lower_lr_from_lower_lr_4s.sh;
# hp for base model
#bash hypopt_gov_rep_100_plus_mask_fix_gemma_it_1b_mt_no_ns_0_8s_all_lin_64_128_new.sh;
# hp for armt model on 2s
bash hypopt_gov_rep_100_plus_mask_fix_armt_gemma_sw_1b_mt_no_ns_0_2s_all_lin_64_128_left_assist_mask_es15_no_sw.sh;
bash hypopt_gov_rep_100_plus_mask_fix_armt_gemma_sw_1b_mt_no_ns_0_2s_all_lin_64_128_left_assist_mask_es15_no_sw_bl_params_from_pretrain_fix.sh;