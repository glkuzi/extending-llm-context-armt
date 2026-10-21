#bash llama_it_8b_mt_no_ns_0_2s.sh;
#bash armt_sw_8b_mt_no_ns_0_2s.sh;
#bash llama_it_8b_mt_no_ns_0_4s.sh;
#bash armt_sw_8b_mt_no_ns_0_4s.sh;
#bash armt_sw_8b_mt_no_ns_0_4s_from_inst.sh;
#bash armt_sw_8b_mt_no_ns_0_2s_from_inst.sh;
# after run 8b 2s from 2s inst and CL on 4s
# and after run CL 2s-4s-8s on 1b model vs 1b model 8s
#bash armt_sw_8b_mt_no_ns_0_4s_from_2s_from_inst.sh;
#bash llama_it_1b_mt_no_ns_0_4s.sh;
#bash armt_sw_1b_mt_no_ns_0_2s.sh;
#bash armt_sw_1b_mt_no_ns_0_4s.sh;
#bash armt_sw_1b_mt_no_ns_0_8s.sh;
#bash armt_sw_1b_mt_no_ns_0_16s.sh;

# fix K2 param
#bash armt_sw_8b_mt_no_ns_0_4s_from_2s_from_inst.sh;
#bash armt_sw_8b_mt_no_ns_0_4s_from_inst.sh;
#bash armt_sw_8b_mt_no_ns_0_4s.sh;
#bash armt_sw_1b_mt_no_ns_0_4s.sh;
#bash armt_sw_1b_mt_no_ns_0_8s.sh;
#bash armt_sw_1b_mt_no_ns_0_16s.sh;
# 1b, 512
#bash llama_it_1b_mt_no_ns_0_8s.sh;
#bash armt_sw_1b_mt_no_ns_512_0_2s.sh;
#bash armt_sw_1b_mt_no_ns_512_0_4s.sh;
#bash armt_sw_1b_mt_no_ns_512_0_8s.sh;
#bash armt_sw_1b_mt_no_ns_512_0_16s.sh;
# run 1b 1024 model with all-linear and full ft
#bash armt_sw_1b_mt_no_ns_0_2s_all_lin_32_128.sh;
#bash armt_sw_1b_mt_no_ns_0_4s_all_lin_32_128.sh;
#bash armt_sw_1b_mt_no_ns_0_8s_all_lin_32_128.sh;
#bash armt_sw_1b_mt_no_ns_0_2s_all_lin_16_128.sh;
#bash armt_sw_1b_mt_no_ns_0_4s_all_lin_16_128.sh;
#bash armt_sw_1b_mt_no_ns_0_8s_all_lin_16_128.sh;
#bash armt_sw_1b_mt_no_ns_0_2s_256_all_lin_64_128.sh;
#bash armt_sw_1b_mt_no_ns_0_4s_256_all_lin_64_128.sh;
#bash armt_sw_1b_mt_no_ns_0_8s_256_all_lin_64_128.sh;
#bash armt_sw_1b_mt_no_ns_0_16s_256_all_lin_64_128.sh;
# broken - continue from last cpt
#bash armt_sw_1b_mt_no_ns_0_32s_256_all_lin_64_128.sh;
#
#bash armt_sw_1b_mt_no_ns_0_2s_all_lin_64_128_comp_256.sh;
#bash armt_sw_1b_mt_no_ns_0_4s_all_lin_64_128_comp_256.sh;
#bash armt_sw_1b_mt_no_ns_0_8s_all_lin_64_128_comp_256.sh;
#bash llama_it_1b_mt_no_ns_0_1s_all_lin_64_128.sh;
#bash armt_sw_1b_mt_no_ns_0_8s_128_all_lin_64_128_reverse_cl.sh;

bash armt_sw_1b_mt_no_ns_0_2s_all_lin_64_128_from_2s_inst.sh;
bash armt_sw_1b_mt_no_ns_0_4s_all_lin_64_128_from_2s_inst.sh;
bash armt_sw_1b_mt_no_ns_0_8s_all_lin_64_128_from_2s_inst.sh;

bash armt_sw_1b_mt_no_ns_0_2s_all_lin_64_128_from_4s_inst_st2s.sh;
bash armt_sw_1b_mt_no_ns_0_4s_all_lin_64_128_from_4s_inst_st2s.sh;
bash armt_sw_1b_mt_no_ns_0_8s_all_lin_64_128_from_4s_inst_st2s.sh;
# OOM - run after on 4 gpus
#bash armt_sw_1b_mt_no_ns_0_16s_all_lin_64_128_comp_256.sh;
#bash armt_sw_1b_mt_no_ns_0_16s_all_lin_64_128_id4.sh;

# run 8b 1024 model with all-linear
