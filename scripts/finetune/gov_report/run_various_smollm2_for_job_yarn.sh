#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
# test labeling for 3-5 additional models to increase data diversity
set -e
ls;
#cd ../../
eval "$(conda shell.bash hook)"
conda activate /workspace-SR006.nfs1/gkuzmin/rmt_it/conda_rmt_it_venv
cd /workspace-SR006.nfs1/gkuzmin/rmt_it/rmt_ift/scripts/finetune/gov_report/


#bash gov_rep_100_plus_mask_fix_smollm2_360m_it_mt_no_ns_0_8s_all_lin_64_128_new_with_yarn_train.sh;
#cd ../manytypes
#bash mt_v2_mask_fix_smollm2_360M_mt_no_ns_0_8s_all_lin_64_128_left_assist_mask_es15_bl_pars.sh;
#cd ../gov_report
# recalc after code fix
#bash gov_rep_100_plus_mask_fix_smollm2_360m_it_mt_no_ns_0_8s_all_lin_64_128_new_with_yarn_train.sh;
#bash gov_rep_100_plus_mask_fix_smollm2_360m_it_mt_no_ns_0_8s_all_lin_64_128_new_with_yarn_pg19_adapt_train.sh;
#bash gov_rep_100_plus_mask_fix_smollm2_360m_it_mt_no_ns_0_8s_all_lin_64_128_new_with_yarn_pg19_adapt_train_1e_4.sh;
bash gov_rep_100_plus_mask_fix_smollm2_360m_it_mt_no_ns_0_8s_all_lin_64_128_new_with_yarn_train_after_adapt.sh;