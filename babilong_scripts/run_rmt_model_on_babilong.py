import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm.auto import tqdm
import pandas as pd
import json
from pathlib import Path
import requests
from safetensors.torch import save_file, load_file
from typing import List
import sys
#sys.path.append('..')
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from modeling_rmt.language_modeling import RecurrentWrapper, MemoryCell
from modeling_amt.language_modeling import AssociativeRecurrentWrapper, AssociativeMemoryCell
from modeling_amt.language_modeling_sw_llama import AssociativeRecurrentWrapper as AssociativeRecurrentWrapperSW
from modeling_amt.language_modeling_sw_llama import AssociativeMemoryCell as AssociativeMemoryCellSW
from modeling_amt.language_modeling_sw_llama import AssociativeLayerWrapperIdentity
from peft import get_peft_model, LoraConfig, TaskType


class DistillWrapper(torch.nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model


def generate(self, input_ids, attention_mask=None, **generate_kwargs):
    """
    Generate tokens using the inner-loop model with proper sliding window attention.
    This method should produce the same logits as the forward method for alignment.
    """

    if self.sliding_window_enabled:
        generate_kwargs["max_new_tokens"] = generate_kwargs["max_length"] - input_ids.shape[-1]
        generate_kwargs["max_length"] = None
        x = self._generate_sliding_window(input_ids, attention_mask, **generate_kwargs)
        return x
    else:
        # pathcing for armt - as we add mem tokens, we shouldn't restrict max_length
        generate_kwargs["max_new_tokens"] = generate_kwargs["max_length"] - input_ids.shape[-1]
        generate_kwargs["max_length"] = None
        x = self._generate_non_sliding_window(input_ids, attention_mask, **generate_kwargs)
        x = torch.cat((input_ids, x), dim=1)
        #print(x)
        #print(generate_kwargs)
        #print(len(input_ids))
        #print(input_ids.shape)
        return x


def main(
    results_folder: str, model_name: str, tasks: List[str], split_names: List[str], dataset_name: str,
    use_chat_template: bool, api_url: str, use_instruction: bool, use_examples: bool, use_post_prompt: bool,
    model_cpt: str, mem_size: int, segment_size: int, max_n_segments: int, lora_r: int,
    lora_alpha: int, lora_dropout: float, use_peft: float, d_mem, layers_attr, no_correction, wrap_pos, use_quest_first_template,
    use_double_question, use_answer_fill, add_question_prompt, segment_alignment, attend_to_previous_input, max_new_tokens,
    add_lora_to_armt, n_heads, use_sink, prev_seg_kv, lora_layers, distill, compress_mem, teacher_name, no_tags_prompt,
    legacy_bl_prompt, lora_ablate, armt_ablate, identity_frequency, non_identity_layers, new_armt,
) -> None:
    """
    Main function to get model predictions on babilong and save them.

    Args:
        results_folder (str): Folder to store results.
        model_name (str): Name of the model to use.
        tasks (List[str]): List of tasks to evaluate.
        split_names (List[str]): List of lengths to evaluate.
        dataset_name (str): Dataset name from Hugging Face.
        use_chat_template (bool): Flag to use the tokenizer chat template.
        api_url (str): API endpoint for llama.cpp.
        use_instruction (bool): Flag to use instruction in prompt.
        use_examples (bool): Flag to use examples in prompt.
        use_post_prompt (bool): Flag to use post_prompt text in prompt.
    """

    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if model_name == "meta-llama/Llama-3.2-1B":
        # hotfix - load 8b instruct tokenizer and reuse chat template from instruct tokenizer
        it_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    elif model_name == "HuggingFaceTB/SmolLM2-135M":
        # hotfix - load 8b instruct tokenizer and reuse chat template from instruct tokenizer
        it_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    elif model_name == "irodkin/ARMT-Gemma-3-1B-IT-8x1024":
        # hotfix - load 8b instruct tokenizer and reuse chat template from instruct tokenizer
        it_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    elif model_name == "irodkin/ARMT-SmolLM2-360m-8x1024":
        # hotfix - load 8b instruct tokenizer and reuse chat template from instruct tokenizer
        it_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    if not api_url:
        if len(model_cpt) != 0:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                         device_map='cpu', torch_dtype=dtype,
                                                         attn_implementation="flash_attention_2" if not prev_seg_kv else "eager")
            # get eos token
            try:
                eos_token_id = model.generation_config.eos_token_id
            except:
                eos_token_id = tokenizer.eos_token_id
            if use_peft:
                lora_layers = lora_layers.split(";") if ";" in lora_layers else lora_layers
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=lora_r, 
                    lora_alpha=lora_alpha, 
                    lora_dropout=lora_dropout,
                    target_modules=lora_layers,
                    )
                model = get_peft_model(model, peft_config)
            device = "cpu"
            if d_mem is not None:
                mem_cell_args = dict(
                    base_model=model,
                    num_mem_tokens=mem_size,
                )
                # additional parameters for ARMT model
                mem_cell_args['d_mem'] = d_mem
                mem_cell_args['wrap_pos'] = wrap_pos
                mem_cell_args['correction'] = not(no_correction)
                mem_cell_args['use_lora'] = use_peft
                if layers_attr is not None:
                    mem_cell_args['layers_attr'] = layers_attr
                if attend_to_previous_input is not None:
                    mem_cell_args['attend_to_previous_input'] = bool(attend_to_previous_input)
                if n_heads is not None:
                    mem_cell_args['n_heads'] = n_heads
                if use_sink:
                    mem_cell_args['use_sink'] = use_sink
                if compress_mem != 0:
                    mem_cell_args['compress_mem'] = compress_mem
                if identity_frequency != 0:
                    mem_cell_args['identity_frequency'] = identity_frequency
                if non_identity_layers is not None:
                    mem_cell_args['non_identity_layers'] = non_identity_layers
                if n_heads is not None:
                    # case of model with sliding window
                    cell = AssociativeMemoryCellSW(**mem_cell_args)
                    model = AssociativeRecurrentWrapperSW(cell, 
                                                          segment_size=segment_size-mem_size,
                                                          max_n_segments=max_n_segments, 
                                                          segment_alignment=segment_alignment,
                                                          sliding_window=prev_seg_kv,
                                                          return_all_logits=False,
                    ).to(device)
                else:
                    cell = AssociativeMemoryCell(**mem_cell_args)
                    model = AssociativeRecurrentWrapper(cell,
                                                        segment_size=segment_size-mem_size,
                                                        max_n_segments=max_n_segments,
                                                        segment_alignment=segment_alignment,
                                                        attend_to_previous_input=attend_to_previous_input,
                    ).to(device)
            elif mem_size is not None:
                cell = MemoryCell(model, num_mem_tokens=mem_size)
                model = RecurrentWrapper(cell,
                                         segment_size=segment_size-2*mem_size,
                                         max_n_segments=max_n_segments,
                ).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_cpt, trust_remote_code=True,
                                                             device_map='cpu', torch_dtype=dtype,
                                                             attn_implementation="flash_attention_2" if not prev_seg_kv else "eager")
            if distill:
                teacher_name = model_name if teacher_name is None else teacher_name
                teacher = AutoModelForCausalLM.from_pretrained(teacher_name, trust_remote_code=True,
                                                               device_map='cpu', torch_dtype=dtype,
                                                               attn_implementation="flash_attention_2" if not prev_seg_kv else "eager")
                model = DistillWrapper(teacher, model)
            #model = model.to(dtype)
            if add_lora_to_armt:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    )
                # add LoRA only to the inner model
                model.memory_cell.model = get_peft_model(model.memory_cell.model, peft_config)
                model.memory_cell.model.print_trainable_parameters()
                print(model)
            if d_mem is not None or mem_size is not None:
                try:
                    cpt = torch.load(model_cpt, map_location=device)
                    model.load_state_dict(cpt, strict=True)
                except:
                    # if the model saved in safetensors
                    from safetensors.torch import load_model
                    load_model(model, model_cpt, device=device)
                    #model.load_state_dict(model_cpt, map_location=device)
            if distill:
                del model.teacher
                torch.cuda.empty_cache()
                model = model.student
            device = 'cuda:0'
            model.to(device)
            model.to(dtype)
            if mem_size is not None:# or "inst_pretrain_armt_sw" in model_name:
                model.name_or_path = "custom_rmt" # workaround
                model.device = device
        # load the model locally if llamacpp API is not used
        else:
            try:
                print('trying to load model with flash attention 2...')
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                             device_map='auto', torch_dtype=dtype,
                                                             attn_implementation="flash_attention_2" if not prev_seg_kv else "eager")
            except ValueError as e:
                print(e)
                print('trying to load model without flash attention 2...')
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                             device_map='auto', torch_dtype=dtype)
            try:
                eos_token_id = model.generation_config.eos_token_id
            except:
                eos_token_id = tokenizer.eos_token_id
        model = model.eval()
    print(lora_ablate)
    if new_armt:
        print(model)
        print(model.model.lm_head.weight)
        import types
        model.generate = types.MethodType(generate, model)
    #print(model)
    model.train()
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model.eval()
    print("Trainable_params:", params)
    reduced_pars = 0
    if lora_ablate is not None:
        #print(model)
        lora_ablate = lora_ablate.split(";")
        for name, par in model.named_parameters():
            if "lora" in name:
                if any([el in name for el in lora_ablate]):
                    #print(name)
                    #print(torch.linalg.matrix_rank(par.data.float()), torch.linalg.norm(par.data.float()))
                    par.data.zero_()
                    reduced_pars += np.prod(par.size())
        #print(smth)
        pass
    if armt_ablate is not None:
        armt_ablate = armt_ablate.split(";")
        for idx in armt_ablate:
            el = int(idx)
            kw = dict(
                layer=model.memory_cell.layers[el].layer, 
                d_model=model.memory_cell.layers[el].d_model, 
                num_mem_tokens=model.memory_cell.layers[el].num_mem_tokens, 
                d_mem=model.memory_cell.layers[el].d_mem,
                correction=model.memory_cell.layers[el].correction,
                info=model.memory_cell.layers[el].info,
                n_heads=model.memory_cell.layers[el].n_heads,
                use_denom=model.memory_cell.layers[el].use_denom,
                gating=model.memory_cell.layers[el].gating,
                compress_mem=model.memory_cell.layers[el].compress_mem
            )
            model.memory_cell.layers[el] = AssociativeLayerWrapperIdentity(**kw)
    #print(model)
    print("Reduced_params:", reduced_pars)
    print("Remaining_params:", params - reduced_pars)
    merge_and_save = False
    unmerge_and_save = False
    if merge_and_save or unmerge_and_save:
        from safetensors.torch import save_model
        print(model)
        if merge_and_save:
            model.memory_cell.model.merge_and_unload()
        if unmerge_and_save:
            model.memory_cell.model.unload()
        print(model)
        new_model_cpt = model_cpt.split(".")
        if merge_and_save:
            new_model_cpt[-2] = new_model_cpt[-2] + "_merged"
        if unmerge_and_save:
            new_model_cpt[-2] = new_model_cpt[-2] + "_unmerged"
        new_model_cpt = ".".join(new_model_cpt)
        model.memory_cell.model = model.memory_cell.model.base_model.model
        print(new_model_cpt)
        print(model)
        save_model(model, new_model_cpt)
        return 0

    # define generation parameters
    generate_kwargs = {
        'max_new_tokens': max_new_tokens if "RMT-Llama-3.2-1B-Instruct-8x1024-mem16-lora-babilong-qa1-5_ct-v3.1" not in model_cpt else 30,
        'max_length': None,
        'num_beams': 1,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'top_k': None,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': eos_token_id
    }

    if generate_kwargs['pad_token_id'] is None:
        generate_kwargs['pad_token_id'] = tokenizer.eos_token_id

    template_to_use = DEFAULT_TEMPLATE
    if use_quest_first_template:
        SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
        USER_TEMPLATE = 'Question: {question}\n\n<context>\n{context}\n</context>'
        if use_double_question:
            USER_TEMPLATE = 'Question: {question}\n\n<context>\n{context}\n</context>\n\nQuestion: {question}'
        template_to_use = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'
    if use_answer_fill:
        template_to_use += '\n\nAnswer: '

    if legacy_bl_prompt:
        SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
        USER_TEMPLATE = '{context} {question} GEN'
        template_to_use = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'

    if no_tags_prompt:
        SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
        USER_TEMPLATE = '{context} {question}'
        template_to_use = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'
    
    print(f'prompt template:\n{template_to_use}')

    for task in tqdm(tasks, desc='tasks'):
        # configure the prompt
        prompt_cfg = {
            'instruction': DEFAULT_PROMPTS[task]['instruction'] if use_instruction else '',
            'examples': DEFAULT_PROMPTS[task]['examples'] if use_examples else '',
            'post_prompt': DEFAULT_PROMPTS[task]['post_prompt'] if use_post_prompt else '',
            'template': template_to_use,
            'chat_template': use_chat_template,
        }
        prompt_name = [f'{k}_yes' if prompt_cfg[k] else f'{k}_no' for k in prompt_cfg if k != 'template']
        prompt_name = '_'.join(prompt_name)
        if use_quest_first_template:
            prompt_name += "_quest_first_yes"
            if use_double_question:
                prompt_name += "_quest_first_2_yes"
        if use_answer_fill:
            prompt_name += "_answer_fill_yes"
        if legacy_bl_prompt:
            prompt_name += "_legacy_bl_mode"
        if no_tags_prompt:
            prompt_name += "_no_tags_mode"
        if lora_ablate is not None:
            print(lora_ablate)
            lora_ablate_str = ";".join(lora_ablate)
            print(lora_ablate_str)
            prompt_name += f"_lora_ablate_{lora_ablate_str}"
        if armt_ablate is not None:
            print(armt_ablate)
            armt_ablate_str = ";".join(armt_ablate)
            print(armt_ablate_str)
            prompt_name += f"_armt_ablate_{armt_ablate_str}"

        for split_name in tqdm(split_names, desc='lengths'):
            # load dataset
            data = datasets.load_dataset(dataset_name, split_name)
            task_data = data[task]

            # Prepare files with predictions, prompt, and generation configurations
            outfile = Path(f'{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/{task}_{split_name}_{prompt_name}.csv')
            outfile.parent.mkdir(parents=True, exist_ok=True)
            print(outfile)
            cfg_file = f'./{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/{task}_{split_name}_{prompt_name}.json'
            json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)

            df = pd.DataFrame({'target': [], 'output': [], 'question': []})

            for sample in tqdm(task_data, desc=f'task: {task} length: {split_name}'):
                target = sample['target']
                context = sample['input']
                question = sample['question']

                # format input text
                input_text = get_formatted_input(context, question + add_question_prompt, prompt_cfg['examples'],
                                                 prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                                 template=prompt_cfg['template'])

                if api_url:
                    # model is running via llamacpp's serve command
                    headers = {'Content-Type': 'application/json'}
                    if generate_kwargs['temperature'] is None:
                        generate_kwargs['temperature'] = 0.0

                    if use_chat_template:
                        input_text = [{'role': 'user', 'content': input_text}]
                        model_inputs = tokenizer.apply_chat_template(input_text, tokenize=True,
                                                                     add_generation_prompt=True)
                    else:
                        model_inputs = tokenizer.encode(input_text, add_special_tokens=True)

                    request_data = {'prompt': model_inputs, 'temperature': generate_kwargs['temperature']}
                    response = requests.post(api_url, headers=headers, json=request_data).json()
                    output = response['content'].strip()
                else:
                    # generate output using local model
                    if model.name_or_path in ['THUDM/chatglm3-6b-128k', 'THUDM/LongAlign-6B-64k-base', 'THUDM/LongAlign-6B-64k']:
                        # have to add special code to run chatglm as tokenizer.chat_template tokenization is not
                        # the same as in model.chat (recommended in https://huggingface.co/THUDM/chatglm3-6b-128k)
                        with torch.no_grad():
                            output, _ = model.chat(tokenizer, input_text, history=[], **generate_kwargs)
                    else:
                        if use_chat_template:
                            input_text = [{'role': 'user', 'content': input_text}]
                            model_inputs = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,
                                                                         return_tensors='pt', return_dict=model.name_or_path=="custom_rmt").to(model.device)
                            if model.name_or_path != "custom_rmt":
                                model_inputs = {'input_ids': model_inputs}
                        else:
                            model_inputs = tokenizer(input_text, return_tensors='pt',
                                                     add_special_tokens=True).to(model.device)

                        sample_length = model_inputs['input_ids'].shape[1]
                        #print(input_text)
                        #print(model_inputs)
                        with torch.no_grad():
                            output = model.generate(**model_inputs, **generate_kwargs)
                            # we need to reset memory states between samples for activation-beacon models
                            if 'activation-beacon' in model.name_or_path and hasattr(model, 'memory'):
                                model.memory.reset()
                        if model.name_or_path != "custom_rmt" and model_name != 'irodkin/ARMT-Gemma-3-1B-IT-8x1024' and model_name != 'irodkin/ARMT-SmolLM2-360m-8x1024':
                            output = output[0][sample_length:]
                        else:
                            #print(generate_kwargs)
                            #print(output[0])
                            output = output[0]
                        output = tokenizer.decode(output, skip_special_tokens=True).strip()
                        #print(output)

                df.loc[len(df)] = [target, output, question]
                # write results to csv file
                df.to_csv(outfile, escapechar='\\')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--results_folder', type=str, required=True, default='./babilong_evals',
                        help='Folder to store results')
    parser.add_argument('--dataset_name', type=str, required=True, default='RMT-team/babilong-1k-samples',
                        help='dataset name from huggingface')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--model_cpt', type=str, default="", help='Name of the model checkpoint in case of RMT')
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help='List of tasks to evaluate: qa1 qa2 ...')
    parser.add_argument('--lengths', type=str, nargs='+', required=True, help='List of lengths to evaluate: 0k 1k ...')
    parser.add_argument('--use_chat_template', action='store_true', help='Use tokenizer chat template')
    parser.add_argument('--use_instruction', action='store_true', help='Use instruction in prompt')
    parser.add_argument('--use_examples', action='store_true', help='Use examples in prompt')
    parser.add_argument('--use_post_prompt', action='store_true', help='Use post prompt text in prompt')
    parser.add_argument('--no_tags_prompt', action='store_true', help='Remove tags from prompt')
    parser.add_argument('--legacy_bl_prompt', action='store_true', help='Use prompt for GPT-2 with GEN token')
    parser.add_argument('--api_url', type=str, required=True, default='', help='llamacpp api endpoint')
    # args for RMT model
    parser.add_argument('--mem_size', type=int, required=False, default=None, help='Memory size for RMT')
    parser.add_argument('--segment_size', type=int, required=False, default=4096, help='Segment size for RMT')
    parser.add_argument('--max_n_segments', type=int, required=False, default=8, help='Max number of segments for RMT')
    # args for ARMT model
    parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
    parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')
    parser.add_argument('--no_correction', action='store_true', default=False,
                        help='ARMT shmidhuber correction for rewriting')
    parser.add_argument('--wrap_pos', action='store_true', default=False,
                        help='Wrap positional encoding for memory tokens (default: False)')
    # LoRA params
    parser.add_argument('--use_peft', type=int, required=False, default=1, help='Use PEFT in model', )
    parser.add_argument('--lora_r', type=int, required=False, default=8, help='LoRA r')
    parser.add_argument('--lora_alpha', type=int, required=False, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, required=False, default=0.1, help='LoRA dropout')
    parser.add_argument('--lora_layers', type=str, default=None, help='')
    parser.add_argument('--lora_ablate', type=str, default=None, help='')
    parser.add_argument('--armt_ablate', type=str, default=None, help='')
    parser.add_argument('--use_quest_first_template', action='store_true', help='Use template with question before context (but after all other parts)')
    parser.add_argument('--use_double_question', action='store_true', help='Use template with question before and after context (but after all other parts)')
    parser.add_argument('--use_answer_fill', action='store_true', help='Use Answer: at the end of prompt')
    parser.add_argument('--add_lora_to_armt', type=int, required=False, default=0, help='Use PEFT for ARMT inner model')
    # new eval params
    parser.add_argument('--add_question_prompt', type=str, default="", help='Additional prompt, injected after question')
    parser.add_argument('--segment_alignment', type=str, default="left", help='Segment alignment for ARMT')
    parser.add_argument('--attend_to_previous_input', type=int, default=0, help='Attend to prev segment')
    parser.add_argument('--max_new_tokens', type=int, default=20, help='Tokens to generate')
    parser.add_argument('--prev_seg_kv', action='store_true', default=False, help='propagate kv from previous segment')
    parser.add_argument('--use_sink', action='store_true', default=False, help='use_attention_sink_token')
    parser.add_argument('--n_heads', type=int, default=None, help='number of heads in associative matrix')
    parser.add_argument('--distill', action='store_true', default=False, help='use distillated model')
    parser.add_argument('--teacher_name', type=str, default=None, help='Name of the teacher model to use')
    parser.add_argument('--compress_mem', type=int, default=0, help='compression size for memory mapping')
    parser.add_argument('--identity_frequency', type=int, default=0, help='frequency of identity associative layers')
    parser.add_argument('--non_identity_layers', type=str, default=None, help='indices of non-identity associative layers in ARMT')
    parser.add_argument('--new_armt', action='store_true', default=False, help='new armt model')


    args = parser.parse_args()
    main(args.results_folder, args.model_name, args.tasks, args.lengths, args.dataset_name, args.use_chat_template,
         args.api_url, args.use_instruction, args.use_examples, args.use_post_prompt, args.model_cpt,
         args.mem_size, args.segment_size, args.max_n_segments, args.lora_r, args.lora_alpha, args.lora_dropout, args.use_peft,
         args.d_mem, args.layers_attr, args.no_correction, args.wrap_pos, args.use_quest_first_template, args.use_double_question,
         args.use_answer_fill, args.add_question_prompt, args.segment_alignment, args.attend_to_previous_input, args.max_new_tokens,
         args.add_lora_to_armt, args.n_heads, args.use_sink, args.prev_seg_kv, args.lora_layers, args.distill,
         args.compress_mem, args.teacher_name, args.no_tags_prompt, args.legacy_bl_prompt, args.lora_ablate, args.armt_ablate,
         args.identity_frequency, args.non_identity_layers, args.new_armt)
