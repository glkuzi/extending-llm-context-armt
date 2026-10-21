from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered", filename="HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json",
                repo_type="dataset", local_dir="../../data/sharegpt_raw_cleaned")
hf_hub_download(repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered", filename="HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json",
                repo_type="dataset", local_dir="../../data/sharegpt_raw_cleaned")