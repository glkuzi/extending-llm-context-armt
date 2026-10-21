# ARMT Fine-Tuning

Code and assets for "Extending LLM Context via Associative Recurrent Memory" EMNLP 2026 paper.

## Datasets

The datasets in paper are based on original GovReport and ManyTypes datasets.

The resampled MT dataset could be accessed via following links for [2k](https://huggingface.co/datasets/AIRI-NLP/manytypes_v2_flat_stacked_no_empty_sv_0k_2k), [4k](https://huggingface.co/datasets/AIRI-NLP/manytypes_v2_flat_stacked_no_empty_sv_0k_4k), [8k](https://huggingface.co/datasets/AIRI-NLP/manytypes_v2_flat_stacked_no_empty_sv_0k_8k_val_16k), and [test set](https://huggingface.co/datasets/AIRI-NLP/test_mt_v2_0k_64k_balnced_support).

The resampled and enlarged GR-100+ dataset could be found via following links for [2k](https://huggingface.co/datasets/AIRI-NLP/gov_report_100_plus_0k_2k), [4k](https://huggingface.co/datasets/AIRI-NLP/gov_report_100_plus_0k_4k), [8k](https://huggingface.co/datasets/AIRI-NLP/gov_report_100_plus_0k_8k), and [test set](https://huggingface.co/datasets/AIRI-NLP/test_gov_rep_0k_64k_balanced_support). There is also an [additional test set for MT](https://huggingface.co/datasets/AIRI-NLP/test_mt_v2_0k_128k_balanced_support), extended up to 128k length.

The other datasets used in paper (BABILong, CNLI) are available in their corresponding repositories.


## Checkpoints

We also released several ARMT checkpoints to ease the future experiments:

1. Original ARMT-Gemma-3-1B-IT checkpoint after continued pre-training on 19B tokens from FineWeb-Edu - https://huggingface.co/irodkin/ARMT-Gemma-3-1B-IT-8x1024-portable
2. ARMT-Gemma-3-1B-IT, trained on MT dataset with curriculum learning - https://huggingface.co/AIRI-NLP/ARMT-Gemma-3-1B-IT-MT-v2-8x1024
3. ARMT-Gemma-3-1B-IT, trained on GR-100+ dataset with curriculum learning - https://huggingface.co/AIRI-NLP/ARMT-Gemma-3-1B-IT-GR100plus-8x1024

## How to run training

The main file is run_finetuning_lm_rmt_hf_trainer.py - it contains the training loop and control overall training process. The scripts for various models training and evaluation are located in [scripts/finetune](scripts/finetune):

1. Directories [scripts/finetune/manytypes](scripts/finetune/manytypes) and [scripts/finetune/gov_report](scripts/finetune/gov_report) contain scripts for main paper results - fine-tuning ARMT with curriculum learning on 2-4-8 segments, and optionally ablating the associative memory. The main scripts are [scripts/finetune/manytypes/run_mt_v2_gemma_best_cl_fix.sh](scripts/finetune/manytypes/run_mt_v2_gemma_best_cl_fix.sh) and [scripts/finetune/gov_report/run_main_exps_gemma_bl_pars.sh](scripts/finetune/gov_report/run_main_exps_gemma_bl_pars.sh).
2. Other directories contain additional ablations for CNLI and BABILong datasets.

To evaluate trained models, please refer to the [evaluation](#evaluation) section.

## Evaluation

Evaluation on BABILong and other tasks - [babilong_scripts](babilong_scripts) - there are several scripts for different models:

1. To evaluate base models, use [babilong_scripts/run_model_on_babilong.py](babilong_scripts/run_model_on_babilong.py). This script follows the same procedure as in BABILong paper; in the same directory one can find .bash scripts examples.
2. To evaluate RMT/ARMT models, use [babilong_scripts/run_rmt_model_on_babilong.py](babilong_scripts/run_rmt_model_on_babilong.py).
3. To evaluate various models on MT or GR-100+ tasks, use [babilong_scripts/run_rmt_model_on_qa.py](babilong_scripts/run_rmt_model_on_qa.py). There are also corresponding bash scripts in the same directory.
4. Finally, after running one of the scripts, one has to extract metrics and build tables from raw generated responses. To do so, use [notebooks/babilong_eval_only_ift.ipynb](notebooks/babilong_eval_only_ift.ipynb) for BABILong, [notebooks/compare_mt_v2_0k_64k_paper.ipynb](notebooks/compare_mt_v2_0k_64k_paper.ipynb) and [notebooks/compare_mt_v2_0k_128k_paper.ipynb](notebooks/compare_mt_v2_0k_128k_paper.ipynb) for MT, [notebooks/compare_gov_rep_paper.ipynb](notebooks/compare_gov_rep_paper.ipynb) for GR-100+.




## References

Current code based on two repositories for RMT:

1. [ [paper](https://arxiv.org/abs/2207.06881) ] [ [code](https://github.com/booydar/recurrent-memory-transformer/) ] **Recurrent Memory Transformer** implementation and various training examples. In the current repo it's used for training RMT-based models.

2. [ [paper](https://arxiv.org/abs/2402.10790) ] [ [code](https://github.com/booydar/babilong) ] **BABILong** - a long-context benchmark that supports 20 tasks and various sources of background text. In the current repo it's used for evaluation models on BABILong.

## How to generate long synthetic dataset

The scripts for labeling additional data for GR-100+ dataset are located in [preprocessing](preprocessing) directory.


## Citation
Version from arxiv:

```
@misc{kuzmin2026extendingllmcontextassociative,
      title={Extending LLM Context via Associative Recurrent Memory}, 
      author={Gleb Kuzmin and Ivan Rodkin and Aydar Bulatov and Yuri Kuratov and Lyudmila Rvanova and Mikhail Katkov and Ilia Sochenkov and Misha Tsodyks and Timothy Baldwin and Mikhail Burtsev and Artem Shelmanov},
      year={2026},
      eprint={2607.11614},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.11614}, 
}
```
