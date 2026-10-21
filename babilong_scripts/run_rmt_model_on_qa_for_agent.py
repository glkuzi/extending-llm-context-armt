import argparse
import numpy as np
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
from modeling_amt.language_modeling_sw_llama import  AssociativeMemoryCell as AssociativeMemoryCellSW
from modeling_amt.language_modeling_sw_llama import AssociativeLayerWrapperIdentity
from peft import get_peft_model, LoraConfig, TaskType
import GPUtil


class DistillWrapper(torch.nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model


CHATGLM_MODELS = frozenset({
    'THUDM/chatglm3-6b-128k',
    'THUDM/LongAlign-6B-64k-base',
    'THUDM/LongAlign-6B-64k',
})


def _extract_sample_fields(
    sample,
    use_cuad_template,
    use_cnli_template,
    truncate_only_context,
    truncate_before,
    tokenizer,
):
    if use_cuad_template:
        if "answers" in sample.keys():
            target = sample['answers']['text'][0] if len(sample['answers']['text']) != 0 else "Not stated."
        else:
            target = sample['answer']
    elif use_cnli_template:
        target = sample['output']
    else:
        target = sample['answer']

    if use_cnli_template:
        context = sample['input']
        question = (
            "Given the following hypothesis and premise, classify it as "
            "'Contradiction', 'Entailment' or 'Not mentioned'. Return only the answer."
        )
    else:
        context = sample['context']
        question = sample['question']

    if truncate_only_context:
        context = tokenizer.decode(tokenizer(context)["input_ids"][truncate_before:])

    return target, question, context


def _pad_encoded_batch(encoded_list, pad_token_id, padding_side='left'):
    max_len = max(enc['input_ids'].shape[-1] for enc in encoded_list)
    batch_size = len(encoded_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, enc in enumerate(encoded_list):
        seq = enc['input_ids'].reshape(-1)
        attn = enc.get('attention_mask')
        if attn is None:
            attn = torch.ones_like(seq)
        else:
            attn = attn.reshape(-1)
        seq_len = seq.shape[0]
        if padding_side == 'left':
            input_ids[i, -seq_len:] = seq
            attention_mask[i, -seq_len:] = attn
        else:
            input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = attn
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def _tokenize_for_generate(
    tokenizer,
    input_texts,
    use_chat_template,
    model,
    truncate_before,
    truncate_only_context,
):
    is_custom_rmt = getattr(model, 'name_or_path', None) == "custom_rmt"
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if use_chat_template:
        if is_custom_rmt:
            encoded_list = []
            for text in input_texts:
                messages = [{'role': 'user', 'content': text}]
                enc = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors='pt',
                    return_dict=True,
                )
                if truncate_before != 0 and not truncate_only_context:
                    enc['input_ids'] = enc['input_ids'][:, truncate_before:]
                    if 'attention_mask' in enc:
                        enc['attention_mask'] = enc['attention_mask'][:, truncate_before:]
                encoded_list.append({k: v.squeeze(0) for k, v in enc.items()})
            model_inputs = _pad_encoded_batch(encoded_list, pad_token_id, padding_side='left')
        else:
            texts = [
                tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for text in input_texts
            ]
            model_inputs = tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                add_special_tokens=False,
            )
            if truncate_before != 0 and not truncate_only_context:
                model_inputs['input_ids'] = model_inputs['input_ids'][:, truncate_before:]
                if 'attention_mask' in model_inputs:
                    model_inputs['attention_mask'] = model_inputs['attention_mask'][:, truncate_before:]
    else:
        model_inputs = tokenizer(
            input_texts,
            return_tensors='pt',
            padding=True,
            add_special_tokens=True,
        )
        if truncate_before != 0 and not truncate_only_context:
            model_inputs['input_ids'] = model_inputs['input_ids'][:, truncate_before:]
            if 'attention_mask' in model_inputs:
                model_inputs['attention_mask'] = model_inputs['attention_mask'][:, truncate_before:]

    if 'attention_mask' not in model_inputs:
        model_inputs['attention_mask'] = (model_inputs['input_ids'] != pad_token_id).long()

    return {key: value.to(model.device) for key, value in model_inputs.items()}#model_inputs.to(model.device)


def _decode_generated_outputs(output, model_inputs, tokenizer, is_custom_rmt):
    decoded = []
    padding_side = 'left'
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs.get('attention_mask')
    for i in range(output.shape[0]):
        if is_custom_rmt:
            gen_ids = output[i]
        else:
            if attention_mask is not None:
                prompt_len = int(attention_mask[i].sum().item())
            else:
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                prompt_len = int((input_ids[i] != pad_token_id).sum().item())
            if padding_side == 'left':
                gen_ids = output[i, len(input_ids[i]):]
            else:
                gen_ids = output[i, prompt_len:]
        #print(len(gen_ids), prompt_len, len(output[i]), len(input_ids[i]))
        decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        #print(decoded[-1])
    return decoded


def _generate_local_outputs(
    model,
    tokenizer,
    input_texts,
    use_chat_template,
    generate_kwargs,
    truncate_before,
    truncate_only_context,
):
    if model.name_or_path in CHATGLM_MODELS:
        outputs = []
        for input_text in input_texts:
            with torch.no_grad():
                output, _ = model.chat(tokenizer, input_text, history=[], **generate_kwargs)
            outputs.append(output.strip() if isinstance(output, str) else output)
        return outputs

    is_custom_rmt = model.name_or_path == "custom_rmt"
    model_inputs = _tokenize_for_generate(
        tokenizer,
        input_texts,
        use_chat_template,
        model,
        truncate_before,
        truncate_only_context,
    )
    with torch.no_grad():
        output = model.generate(**model_inputs, **generate_kwargs)
        if 'activation-beacon' in model.name_or_path and hasattr(model, 'memory'):
            model.memory.reset()
    return _decode_generated_outputs(output, model_inputs, tokenizer, is_custom_rmt)


def _generate_api_outputs(api_url, tokenizer, input_texts, use_chat_template, generate_kwargs):
    headers = {'Content-Type': 'application/json'}
    temperature = generate_kwargs['temperature']
    if temperature is None:
        temperature = 0.0

    outputs = []
    for input_text in input_texts:
        if use_chat_template:
            messages = [{'role': 'user', 'content': input_text}]
            model_inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
            )
        else:
            model_inputs = tokenizer.encode(input_text, add_special_tokens=True)

        request_data = {'prompt': model_inputs, 'temperature': temperature}
        response = requests.post(api_url, headers=headers, json=request_data).json()
        outputs.append(response['content'].strip())
    return outputs


def main(
    results_folder: str, model_name: str, tasks: List[str], split_names: List[str], dataset_name: str, dataset_samples: int,
    use_chat_template: bool, api_url: str, use_instruction: bool, use_examples: bool, use_post_prompt: bool,
    model_cpt: str, mem_size: int, segment_size: int, max_n_segments: int, lora_r: int,
    lora_alpha: int, lora_dropout: float, use_peft: float, d_mem, layers_attr, no_correction, wrap_pos, use_quest_first_template,
    use_double_question, use_answer_fill, add_question_prompt, pre_question_prompt, segment_alignment, attend_to_previous_input,
    max_new_tokens, truncate_before, truncate_only_context, n_heads, use_sink, prev_seg_kv, lora_layers, distill, compress_mem,
    teacher_name, use_cuad_template, identity_frequency, non_identity_layers, override_attn, armt_ablate,
    use_yarn, yarn_original_max_position_embeddings, yarn_scaling_type,
    yarn_scaling_factor, yarn_rope_theta, yarn_sliding_window_attention_schedule,
    yarn_max_position_embeddings, use_armt_pretrain, use_cnli_template, bs,
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

    print(model_name)
    print(model_cpt)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if model_name == "meta-llama/Llama-3.2-1B":
        # hotfix - load 8b instruct tokenizer and reuse chat template from instruct tokenizer
        it_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    if not api_url:
        if len(model_cpt) != 0:
            attn_implementation="flash_attention_2" if "gemma" not in model_cpt else "eager"
            overall_attn_implementation = attn_implementation if not prev_seg_kv else "eager"
            if override_attn is not None:
                overall_attn_implementation = override_attn
            if use_yarn:
                from lm_experiments_tools.modeling_llama_yarn import LlamaForCausalLM
                from lm_experiments_tools.configuration_llama import LlamaConfig
                config_cls = LlamaConfig
                model_cls = LlamaForCausalLM
                original_max_position_embeddings = yarn_original_max_position_embeddings if yarn_original_max_position_embeddings else 8192
                config = config_cls.from_pretrained(model_name)
                config.rope_scaling = {
                    "type": yarn_scaling_type,
                    "factor": yarn_scaling_factor,
                    "original_max_position_embeddings": original_max_position_embeddings
                }
                config.rope_theta = yarn_rope_theta
                config.max_position_embeddings = int(yarn_scaling_factor * original_max_position_embeddings) \
                    if not yarn_max_position_embeddings else yarn_max_position_embeddings

                sliding_window_attention_schedule = [int(x) for x in yarn_sliding_window_attention_schedule.split(",")] \
                    if yarn_sliding_window_attention_schedule else None
                if sliding_window_attention_schedule is not None and len(sliding_window_attention_schedule) == 1:
                    config.sliding_window = sliding_window_attention_schedule[0]
                model = model_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    config=config,
                    attn_implementation=attn_implementation,
                )
            elif use_armt_pretrain:
                model = AutoModelForCausalLM.from_pretrained(use_armt_pretrain,
                                                             trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                             device_map='cpu', torch_dtype=dtype,
                                                             attn_implementation=overall_attn_implementation)
            try:
                eos_token_id = model.generation_config.eos_token_id
            except:
                eos_token_id = tokenizer.eos_token_id
            if use_peft:
                if lora_layers == "all-linear" and use_armt_pretrain:
                    exclude_armt = {"W_mq", "W_mk", "W_mv", "W_mb", "lm_head"}
                    try:
                        from transformers.pytorch_utils import Conv1D
                    except Exception:
                        Conv1D = ()
                    lora_layers = sorted({
                        name.split(".")[-1]
                        for name, module in model.named_modules()
                        if isinstance(module, (torch.nn.Linear, Conv1D))
                        and name.split(".")[-1] not in exclude_armt
                    })
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
                pass
            #model = model.to(dtype)
            if distill:
                teacher_name = model_name if teacher_name is None else teacher_name
                teacher = AutoModelForCausalLM.from_pretrained(teacher_name, trust_remote_code=True,
                                                               device_map='cpu', torch_dtype=dtype,
                                                               attn_implementation=overall_attn_implementation)
                model = DistillWrapper(teacher, model)
            if mem_size is None and d_mem is None and use_peft and not use_armt_pretrain:
                # load adapter for model w/o ARMT
                attn_implementation="flash_attention_2" if "gemma" not in model_cpt else "flash_attention_2"
                overall_attn_implementation=attn_implementation if not prev_seg_kv else "eager"
                if override_attn is not None:
                    overall_attn_implementation = override_attn
                model = AutoModelForCausalLM.from_pretrained(model_cpt, trust_remote_code=True,
                                                             device_map='cpu', torch_dtype=dtype,
                                                             attn_implementation=overall_attn_implementation)
            else:
                try:
                    print(model)
                    cpt = torch.load(model_cpt, map_location=device)
                    if use_armt_pretrain:
                        model.load_state_dict(cpt, strict=True)
                    else:
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
            
            #model.name_or_path = "custom_rmt" # workaround
            if mem_size is not None:
                model.name_or_path = "custom_rmt" # workaround
                model.device = device
            #model.device = device
        # load the model locally if llamacpp API is not used
        else:
            if "qwen" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                             device_map='auto', torch_dtype="auto")
            else:
                try:
                    print('trying to load model with flash attention 2...')
                    attn_implementation="flash_attention_2"
                    overall_attn_implementation = attn_implementation if not prev_seg_kv else "eager"
                    if override_attn is not None:
                        overall_attn_implementation = override_attn
                    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                                device_map='auto', torch_dtype=dtype,
                                                                attn_implementation=overall_attn_implementation)
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

    # define generation parameters
    generate_kwargs = {
        'max_new_tokens': max_new_tokens,
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
    if use_cuad_template:
        SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
        USER_TEMPLATE = 'Question:\n{question}\nContext:\n{context}\n'
        template_to_use = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'
        if not use_quest_first_template:
            SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
            USER_TEMPLATE = 'Context:\n{context}\nQuestion:\n{question}\n'
            template_to_use = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'
    if use_cnli_template:
        #SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
        USER_TEMPLATE = 'Instruction:\n{question}\nHypothesis and premise:\n{context}\n'
        template_to_use = f'{USER_TEMPLATE}'
    print(f'prompt template:\n{template_to_use}')

    # setting some default task and length - for now, it doesn't used
    # NB! eval only in question-first + chat template setting, without any additions
    task = "qa1"
    split_name = "0k"
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
    if armt_ablate is not None:
            print(armt_ablate)
            armt_ablate_str = ";".join(armt_ablate)
            print(armt_ablate_str)
            prompt_name += f"_armt_ablate_{armt_ablate_str}"

    # load dataset
    data = datasets.load_from_disk(dataset_name)
    if "cuad" in dataset_name:
        data = data["test"]
    task_data = data.select(np.arange(dataset_samples)) if dataset_samples != -1 else data

    # Prepare files with predictions, prompt, and generation configurations
    pre_quest = "pre_quest_yes" if pre_question_prompt != "" else ""
    post_quest = "post_quest_yes" if add_question_prompt != "" else ""
    outfile = Path(f'{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/qa_test_{pre_quest}_{post_quest}_{task}_{split_name}_{prompt_name}.csv')
    print(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg_file = f'./{results_folder}/{model_name.replace("../", "")}/{model_cpt.replace("../", "")}/qa_test_{pre_quest}_{post_quest}_{task}_{split_name}_{prompt_name}.json'
    if truncate_only_context:
        outfile = str(outfile)[:-4] + f"_trunc_cont_{truncate_before}.csv"
        cfg_file = cfg_file[:-5] + f"_trunc_cont_{truncate_before}.json"
    elif truncate_before != 0:
        outfile = str(outfile)[:-4] + f"_trunc_{truncate_before}.csv"
        cfg_file = cfg_file[:-5] + f"_trunc_{truncate_before}.json"
    if override_attn is not None:
        outfile = str(outfile)[:-4] + f"_override_attn_{override_attn}.csv"
        cfg_file = cfg_file[:-5] + f"_override_attn_{override_attn}.json"
    if bs >= 1:
        outfile = str(outfile)[:-4] + f"_bs{bs}.csv"
        cfg_file = cfg_file[:-5] + f"_bs{bs}.json"

    run_cfg = {
        'prompt': prompt_cfg,
        'generate_kwargs': generate_kwargs,
        'batch_size': bs,
    }
    json.dump(run_cfg, open(cfg_file, 'w'), indent=4)

    if not api_url:
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    df = pd.DataFrame({'target': [], 'output': [], 'question': [], 'context': [], 'time': [], 'mem_usage': []})
    GPUs = GPUtil.getGPUs()
    print(GPUs)
    import time
    start = time.time()

    batch_starts = range(0, len(task_data), bs)
    for batch_start in tqdm(batch_starts, desc=f'task: {task} length: {split_name} bs={bs}'):
        batch_end = min(batch_start + bs, len(task_data))
        batch_rows = []
        input_texts = []
        for sample_idx in range(batch_start, batch_end):
            sample = task_data[sample_idx]
            target, question, context = _extract_sample_fields(
                sample,
                use_cuad_template,
                use_cnli_template,
                truncate_only_context,
                truncate_before,
                tokenizer,
            )
            input_text = get_formatted_input(
                context,
                pre_question_prompt + question + add_question_prompt,
                prompt_cfg['examples'],
                prompt_cfg['instruction'],
                prompt_cfg['post_prompt'],
                template=prompt_cfg['template'],
            )
            batch_rows.append((target, question, context))
            input_texts.append(input_text)

        start_sample = time.time()
        if api_url:
            outputs = _generate_api_outputs(
                api_url, tokenizer, input_texts, use_chat_template, generate_kwargs,
            )
        else:
            outputs = _generate_local_outputs(
                model,
                tokenizer,
                input_texts,
                use_chat_template,
                generate_kwargs,
                truncate_before,
                truncate_only_context,
            )
        end_sample = time.time()
        batch_time = (end_sample - start_sample) / len(batch_rows)
        GPUs = GPUtil.getGPUs()
        mem_usage = GPUs[2].memoryUsed if len(GPUs) > 1 else (GPUs[0].memoryUsed if GPUs else None)
        torch.cuda.empty_cache()

        for (target, question, context), output in zip(batch_rows, outputs):
            df.loc[len(df)] = [target, output, question, context, batch_time, mem_usage]
        df.to_csv(outfile, escapechar='\\')

    end = time.time()
    run_cfg['inference_time'] = end - start
    json.dump(run_cfg, open(cfg_file, 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--results_folder', type=str, required=True, default='./babilong_evals',
                        help='Folder to store results')
    parser.add_argument('--dataset_name', type=str, required=True, default='RMT-team/babilong-1k-samples',
                        help='dataset name from huggingface')
    parser.add_argument('--dataset_samples', type=int, required=False, default=-1,
                        help='First-N samples from dataset used for evaluation')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--model_cpt', type=str, default="", help='Name of the model checkpoint in case of RMT')
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help='List of tasks to evaluate: qa1 qa2 ...')
    parser.add_argument('--lengths', type=str, nargs='+', required=True, help='List of lengths to evaluate: 0k 1k ...')
    parser.add_argument('--use_chat_template', action='store_true', help='Use tokenizer chat template')
    parser.add_argument('--use_instruction', action='store_true', help='Use instruction in prompt')
    parser.add_argument('--use_examples', action='store_true', help='Use examples in prompt')
    parser.add_argument('--use_post_prompt', action='store_true', help='Use post prompt text in prompt')
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
    parser.add_argument('--use_quest_first_template', action='store_true', help='Use template with question before context (but after all other parts)')
    parser.add_argument('--use_double_question', action='store_true', help='Use template with question before and after context (but after all other parts)')
    parser.add_argument('--use_answer_fill', action='store_true', help='Use Answer: at the end of prompt')
    parser.add_argument('--use_cuad_template', action='store_true', help='Use template for cuad dataset')
    # new eval params
    parser.add_argument('--add_question_prompt', type=str, default="", help='Additional prompt, injected after question')
    parser.add_argument('--pre_question_prompt', type=str, default="", help='Additional prompt, injected before question')
    parser.add_argument('--segment_alignment', type=str, default="left", help='Segment alignment for ARMT')
    parser.add_argument('--attend_to_previous_input', type=int, default=0, help='Attend to prev segment')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Tokens to generate')
    parser.add_argument('--truncate_before', type=int, default=0, help='Truncate first N tokens, for debug')
    parser.add_argument('--truncate_only_context', action='store_true', help='Truncate first N tokens only from context')
    parser.add_argument('--prev_seg_kv', action='store_true', default=False, help='propagate kv from previous segment')
    parser.add_argument('--use_sink', action='store_true', default=False, help='use_attention_sink_token')
    parser.add_argument('--n_heads', type=int, default=None, help='number of heads in associative matrix')
    parser.add_argument('--distill', action='store_true', default=False, help='use distillated model')
    parser.add_argument('--teacher_name', type=str, default=None, help='Name of the teacher model to use')
    parser.add_argument('--compress_mem', type=int, default=0, help='compression size for memory mapping')
    parser.add_argument('--identity_frequency', type=int, default=0, help='frequency for identity associative layers')
    parser.add_argument('--non_identity_layers', type=str, default=None, help='indices of non-identity associative layers in ARMT')
    parser.add_argument('--override_attn', type=str, default=None, help='set attention implementation for model, override base value')
    parser.add_argument('--armt_ablate', type=str, default=None, help='')
    # yarn params
    parser.add_argument('--use_yarn', action='store_true', default=False,
                        help='Train base model with YARN context extension; only for Llama-like')
    parser.add_argument('--yarn_original_max_position_embeddings', type=int, default=8192, help='')
    parser.add_argument('--yarn_scaling_type', type=str, default="yarn")
    parser.add_argument('--yarn_scaling_factor', type=float, default=16.0)
    parser.add_argument('--yarn_rope_theta', type=float, default=100000.0)
    parser.add_argument('--yarn_sliding_window_attention_schedule', type=str)
    parser.add_argument('--yarn_max_position_embeddings', type=int)
    # new armt params
    parser.add_argument('--use_armt_pretrain', type=str,
                        help='Train ARMT with inner implementation from LM pretrain')
    parser.add_argument('--use_cnli_template', action='store_true', help='Use template for CNLI dataset')
    parser.add_argument('--bs', type=int, required=False, default=1, help='Batch size')



    args = parser.parse_args()
    main(args.results_folder, args.model_name, args.tasks, args.lengths, args.dataset_name, args.dataset_samples, args.use_chat_template,
         args.api_url, args.use_instruction, args.use_examples, args.use_post_prompt, args.model_cpt,
         args.mem_size, args.segment_size, args.max_n_segments, args.lora_r, args.lora_alpha, args.lora_dropout, args.use_peft,
         args.d_mem, args.layers_attr, args.no_correction, args.wrap_pos, args.use_quest_first_template, args.use_double_question,
         args.use_answer_fill, args.add_question_prompt, args.pre_question_prompt, args.segment_alignment, args.attend_to_previous_input,
         args.max_new_tokens, args.truncate_before, args.truncate_only_context, args.n_heads, args.use_sink, args.prev_seg_kv,
         args.lora_layers, args.distill, args.compress_mem, args.teacher_name, args.use_cuad_template,
         args.identity_frequency, args.non_identity_layers, args.override_attn, args.armt_ablate,
         args.use_yarn, args.yarn_original_max_position_embeddings, args.yarn_scaling_type,
         args.yarn_scaling_factor, args.yarn_rope_theta, args.yarn_sliding_window_attention_schedule,
         args.yarn_max_position_embeddings, args.use_armt_pretrain, args.use_cnli_template, args.bs)
