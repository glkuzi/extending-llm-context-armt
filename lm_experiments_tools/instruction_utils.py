import numpy as np
import warnings


def formatting_func_flan_it(example, tokenizer):
    messages = [
        {"role": "user", "content": example['inputs']},
        {"role": "assistant", "content": example['targets']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_flan(example):
    input_prompt = (f"{example['inputs']}\n\n"
        f"{example['targets']}")
    return {"text" : input_prompt}


def formatting_func_dolly_it(example, tokenizer):
    if example.get("context", "") != "":
        messages = [
            {"role": "user", "content": example['instruction'] + "\n" + example['context']},
            {"role": "assistant", "content": example['response']}
        ]
    else:
        messages = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['response']}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_synth_it(example, tokenizer):
    messages = [
        {"role": "user", "content": example['question'] + "\n" + example['context']},
        {"role": "assistant", "content": example['answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_synth_it_debug(example, tokenizer):
    messages = [
        {"role": "user", "content": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\nInstruction:\n{example['question']}\n\nContext:\n{example['context']}"},
        {"role": "assistant", "content": example['answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_synth_it_verb_inst(example, tokenizer):
    messages = [
        {"role": "user", "content": f"You will be given a question and a context. You must answer the question based on the information in the context. Your response should contain three parts: a recollection of the question, step-by-step thoughts leading to the answer, and an answer to the question.\n {example['question']}+\n{example['context']}"},
        {"role": "assistant", "content": example['answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_synth_it_inst(example, tokenizer):
    messages = [
        {"role": "user", "content": example['instruction'] + "\n\n" + example['question'] + "\n\n" + example['context']},
        {"role": "assistant", "content": example['answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_dolly(example):
    if example.get("context", "") != "":
        input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Input: \n"
        f"{example['context']}\n\n"
        f"### Response: \n"
        f"{example['response']}")
    else:
        input_prompt = (f"Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}")
    return {"text" : input_prompt}


def formatting_func_unnat_it(example, tokenizer):
    if "None" not in example['instances'][0]["constraints"]:
        messages = [
            {"role": "user", "content": example['instances'][0]['instruction_with_input'] + example['instances'][0]['constraints']},
            {"role": "assistant", "content": example['instances'][0]['output']}
        ]
    else:
        messages = [
            {"role": "user", "content": example['instances'][0]['instruction_with_input']},
            {"role": "assistant", "content": example['instances'][0]['output']}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}

def formatting_func_unnat(example):
    input_prompt = (f"{example['instances'][0]['instruction_with_input']} + {example['instances'][0]['constraints'] if 'None' not in example['instances'][0]['constraints'] else ''}\n\n"
        f"{example['instances'][0]['output']}")
    return {"text" : input_prompt}

def formatting_func_la_it(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_dfq_golden_it(example, tokenizer):
    messages = [
        {"role": "user", "content": example['gold_evidence_text'] + "\n" + example['Question']},
        {"role": "assistant", "content": example['Answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_cuad_it(example, tokenizer):
    if len(example['answers']['text']) != 0:
        answer = example['answers']['text'][0]
    else:
        answer = "Not stated."
    messages = [
        {"role": "user", "content": f"Question:\n{example['question']}\nContext:\n{example['context']}\n"},
        {"role": "assistant", "content": f"Answer:\n{answer}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_mt_it(example, tokenizer):
    messages = [
        {"role": "user", "content": f"Question:\n{example['question']}\nContext:\n{example['context']}\n"},
        {"role": "assistant", "content": f"Answer:\n{example['answer']}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_mt_it_end(example, tokenizer):
    messages = [
        {"role": "user", "content": f"Context:\n{example['context']}\nQuestion:\n{example['question']}\n"},
        {"role": "assistant", "content": f"Answer:\n{example['answer']}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_cuad_it_end(example, tokenizer):
    if len(example['answers']['text']) != 0:
        answer = example['answers']['text'][0]
    else:
        answer = "Not stated."
    messages = [
        {"role": "user", "content": f"Context:\n{example['context']}\nQuestion:\n{example['question']}\n"},
        {"role": "assistant", "content": f"Answer:\n{answer}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_cuad_it_recall(example, tokenizer):
    if len(example['answers']['text']) != 0:
        answer = example['answers']['text'][0]
    else:
        answer = "Not stated."
    messages = [
        {"role": "user", "content": f"Question:\n{example['question']}\nContext:\n{example['context']}\n"},
        {"role": "assistant", "content": f"Question:\n{example['question']}\nAnswer:\n{answer}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_cuad_it_inst(example, tokenizer):
    if len(example['answers']['text']) != 0:
        answer = example['answers']['text'][0]
    else:
        answer = "Not stated."
    #instruction = "You will be given question and context. Your task is to give an answer to question based on context. Answer should contain text span."
    instruction = "Given the following context and question, extract the shortest contiguous text span that directly answers the question. Return only the exact words from the context without any additional explanation or generation."
    messages = [
        {"role": "user", "content": f"{instruction}\nQuestion:\n{example['question']}\nContext:\n{example['context']}\n"},
        {"role": "assistant", "content": f"Answer:\n{answer}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_cuad_it_inst_recall(example, tokenizer):
    if len(example['answers']['text']) != 0:
        answer = example['answers']['text'][0]
    else:
        answer = "Not stated."
    #instruction = "You will be given question and context. Your task is to give an answer to question based on context. Answer should contain text span."
    instruction = "Given the following context and question, extract the shortest contiguous text span that directly answers the question. Return the original question and the answer with exact words from the context without any additional explanation or generation."
    messages = [
        {"role": "user", "content": f"{instruction}\nQuestion:\n{example['question']}\nContext:\n{example['context']}\n"},
        {"role": "assistant", "content": f"Question:\n{example['question']}\nAnswer:\n{answer}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_cnli_it(example, tokenizer):
    instruction = "Given the following hypothesis and premise, classify it as 'Contradiction', 'Entailment' or 'Not mentioned'. Return only the answer."
    messages = [
        {"role": "user", "content": f"Instruction:\n{instruction}\nHypothesis and premise:\n{example['input']}\n"},
        {"role": "assistant", "content": f"Answer:\n{example['output']}\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_fineweb_debug(example, tokenizer):
    return {"text" : example['text']}


def mask_non_completion(input, response_template, tokenizer, ignore_index=-100):
    # adapted from huggingface DataCollatorForCompletionOnlyLM https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L87
    if isinstance(response_template, str):
        # The user provides a string, must tokenize
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    else:
        # The user already provides the token ids
        response_token_ids = response_template
    response_token_ids_start_idx = None
    output = np.array(input.copy())
    for idx in np.where(output == response_token_ids[0])[0]:
        # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
        if (
            response_token_ids
            == output[idx : idx + len(response_token_ids)].tolist()
        ):
            response_token_ids_start_idx = idx

    if response_token_ids_start_idx is None:
        warnings.warn(
            f"Could not find response key `{response_template}` in the "
            f'following instance: {tokenizer.decode(output)} '
            f"This instance will be ignored in loss calculation. "
            f"Note, if this happens often, consider increasing the `max_seq_length`."
        )
        output[:] = ignore_index
    else:
        response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

        # Make pytorch loss function ignore all tokens up through the end of the response key
        output[:response_token_ids_end_idx] = ignore_index
    return output.tolist()


def mask_non_completion_multi(input, response_template, end_response_token_ids, tokenizer, ignore_index=-100, mask_type="base"):
    # adapted from huggingface DataCollatorForCompletionOnlyLM https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L87
    if isinstance(response_template, str):
        # The user provides a string, must tokenize
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    else:
        # The user already provides the token ids
        response_token_ids = response_template
    response_token_ids_start_idx = None
    response_token_ids_start_ids = []
    output = np.array(input.copy())
    end_response_token_start_ids = np.where(output == end_response_token_ids)[0]
    for idx in np.where(output == response_token_ids[0])[0]:
        # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
        if (
            response_token_ids
            == output[idx : idx + len(response_token_ids)].tolist()
        ):
            response_token_ids_start_idx = idx
            response_token_ids_start_ids.append(response_token_ids_start_idx)
            #print(idx)

    #end_response_token_start_ids = np.where(output == end_response_token_ids)[0]
    #print(end_response_token_start_ids)
    if response_token_ids_start_idx is None:
        warnings.warn(
            f"Could not find response key `{response_template}` in the "
            f'following instance: {tokenizer.decode(output)} '
            f"This instance will be ignored in loss calculation. "
            f"Note, if this happens often, consider increasing the `max_seq_length`."
        )
        output[:] = ignore_index
    else:
        if len(response_token_ids_start_ids) == 1:
            # case of one reply from assistant
            if mask_type == "base":
                response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)
            elif mask_type == "assistant":
                response_token_ids_end_idx = response_token_ids_start_idx
            output[:response_token_ids_end_idx] = ignore_index
        else:
            # case of multiple step conversation
            if mask_type == "base":
                response_token_ids_start_ids = [el + len(response_token_ids) for el in response_token_ids_start_ids]
            elif mask_type == "assistant":
                response_token_ids_start_ids = [el for el in response_token_ids_start_ids]
            response_token_ids_end_ids = []
            for el in response_token_ids_start_ids:
                buf_end_toks = end_response_token_start_ids - el
                closest_end_idx = end_response_token_start_ids[np.where(buf_end_toks > 0)[0][np.argmin((buf_end_toks)[np.where(buf_end_toks > 0)])]]
                #print(buf_end_toks)
                response_token_ids_end_ids.append(closest_end_idx)
            #print(response_token_ids_start_ids)
            #print(response_token_ids_end_ids)
            #response_token_ids_start_ids = [el + len(response_token_ids) for el in response_token_ids_start_ids]
            mask = np.zeros_like(output)
            for response_idx in range(len(response_token_ids_start_ids)):
                start_idx = response_token_ids_start_ids[response_idx]
                try:
                    end_idx = response_token_ids_end_ids[response_idx]
                except:
                    end_idx = -1
                mask[start_idx:end_idx] = 1
            output[np.where(mask == 0)[0]] = ignore_index

        # Make pytorch loss function ignore all tokens up through the end of the response key
        #output[:response_token_ids_end_idx] = ignore_index
    #print(input)
    #print(output)
    #print(smth)
    return output.tolist()


def formatting_func_dispatcher(task_name):
    if "flan" in task_name:
        return formatting_func_flan_it
    elif "dolly" in task_name or "alpaca" in task_name:
        return formatting_func_dolly_it
    elif "LongAlign" in task_name or "LAMix" in task_name or "sharegpt" in task_name or "ultrachat" in task_name or "smoltalk" in task_name:
        return formatting_func_la_it
    elif "unnatural" in task_name:
        return formatting_func_unnat_it
    elif "synth_reduced_1k_2k_debug" in task_name:
        return formatting_func_synth_it_debug
    elif "synth_reduced_cot_1k_2k_verb_inst" in task_name:
        return formatting_func_synth_it_verb_inst
    elif "synth_reduced_qa_inst" in task_name:
        return formatting_func_synth_it_inst
    elif "synth" in task_name:
        return formatting_func_synth_it
    elif "docfinqa_golden" in task_name:
        return formatting_func_dfq_golden_it
    elif "cuad_no_ns_inst" in task_name:
        return formatting_func_cuad_it_inst
    elif "cuad_no_ns_quest_recall" in task_name:
        return formatting_func_cuad_it_recall
    elif "cuad_no_ns_quest_end" in task_name:
        return formatting_func_cuad_it_end
    elif "cuad_no_ns_inst_recall" in task_name:
        return formatting_func_cuad_it_inst_recall
    elif "cuad" in task_name:
        return formatting_func_cuad_it
    elif "manytypes_quest_end" in task_name:
        return formatting_func_mt_it_end
    elif "manytypes" in task_name:
        return formatting_func_mt_it
    elif "gov_report" in task_name:
        return formatting_func_mt_it
    elif "fineweb_train_subset" in task_name or "pg19_subset_yarn" in task_name:
        return formatting_func_fineweb_debug
    elif "cnli" in task_name:
        return formatting_func_cnli_it
    else:
        raise NotImplementedError(f"There is no formatting function for task {task_name}!")

def formatting_func_dispatcher_non_it(task_name):
    if "flan" in task_name:
        return formatting_func_flan
    elif "dolly" in task_name or "alpaca" in task_name:
        return formatting_func_dolly
    elif "LongAlign" in task_name or "LAMix" in task_name or "sharegpt" in task_name or "ultrachat" in task_name:
        return None
    elif "unnatural" in task_name:
        return formatting_func_unnat
    else:
        raise NotImplementedError(f"There is no formatting function for task {task_name}!")
