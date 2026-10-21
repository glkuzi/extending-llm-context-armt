# gemma on base gr
#bash gemma_it_1b_mt_no_ns_0_8s_all_lin_64_128.sh;
bash gemma_it_1b_mt_no_ns_0_8s_full_ft.sh;
bash armt_gemma_1b_mt_no_ns_0_2s_all_lin_64_128_no_sw.sh;
bash armt_gemma_1b_mt_no_ns_0_4s_all_lin_64_128_no_sw.sh;
bash armt_gemma_1b_mt_no_ns_0_8s_all_lin_64_128_no_sw.sh;
# gemma on 50 gen gr
bash gemma_it_1b_mt_no_ns_gen_50_0_8s_all_lin_64_128.sh;
bash gemma_it_1b_mt_no_ns_gen_50_0_8s_full_ft.sh;
bash armt_gemma_1b_mt_no_ns_gen_50_0_2s_all_lin_64_128_no_sw.sh;
bash armt_gemma_1b_mt_no_ns_gen_50_0_2s_all_lin_64_128_no_sw.sh;
bash armt_gemma_1b_mt_no_ns_gen_50_0_2s_all_lin_64_128_no_sw.sh;
# after run manytypes with id4
cd ../manytypes
bash run_various_2s_gemma.sh;
cd ../gov_report
bash armt_gemma_1b_mt_no_ns_gen_50_0_4s_all_lin_64_128_no_sw.sh;
bash armt_gemma_1b_mt_no_ns_gen_50_0_8s_all_lin_64_128_no_sw.sh;
# Now run llama with left pad and assist mask on govrep
bash llama_it_1b_mt_no_ns_0_8s_all_lin_64_128_left_pad_assist_mask.sh;
bash armt_sw_1b_mt_no_ns_0_2s_all_lin_64_128_left_assist_mask.sh;
bash armt_sw_1b_mt_no_ns_0_4s_all_lin_64_128_left_assist_mask.sh;
bash armt_sw_1b_mt_no_ns_0_8s_all_lin_64_128_left_assist_mask.sh;
# TODO: after try to train gemma with eager attn