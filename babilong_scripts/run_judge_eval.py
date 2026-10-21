import datasets
import argparse
import os
from openai import OpenAI
import pandas as pd
import numpy as np
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, GenerationConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt, TextPrompt
import httpx


instruction_score_only_lambda = lambda x: f"""You are an expert evaluator tasked with comparing two answers to a question based on a given context. Your goal is to assess how similar the second answer is to the ground truth answer, with respect to the question and the context provided. Follow these steps:

    Understand the context: Carefully read the provided text to fully grasp the background information relevant to the question.

    Analyze the question: Identify the key points or requirements of the question.

    Evaluate the ground truth answer: Understand why the ground truth answer is correct and how it aligns with the question and context.

    Compare the second answer: Assess the second answer by analyzing its content, relevance, and accuracy in relation to the ground truth, question, and context.

    Provide a similarity score: Rate the similarity between the second answer and the ground truth on a scale of 0 to 10, where 0 means no similarity and 10 means identical in meaning and relevance.

    The final sentence should contain only the similarity score in the following format: ''Similarity Score: Your Score/10''

Input:

    Context: {x['context']}

    Question: {x['question']}

    Ground Truth Answer: {x['target']}

    Answer to Check: {x['output']}

Output:

    Similarity Score: [Your score from 0 to 10, formatted as ''Similarity Score: Your Score/10'']
    The final sentence should contain only the similarity score in the following format: ''Similarity Score: Your Score/10''

"""


prompt_from_hf_judge_base_prompt =  lambda x: f'''I require an evaluation of a model's generated answer against the ground-truth answer. I'll provide you with context, a question, the ground-truth answer, and the model's generated answer. Your task is to assess how well the generated answer matches the ground-truth from a human perspective.

## Input Data

{{
    "context": """{x['context']}""",
    "question": """{x['question']}""",
    "ground_truth": """{x['target']}""",
    "generated_answer": """{x['output']}"""
}}

## Task

Evaluate the generated answer based on how accurately it matches the ground-truth answer in terms of:
1. Factual correctness
2. Completeness of information
3. Relevance to the question
4. Clarity and coherence

Provide your evaluation as a score from 0 to 10, where:
- 0-2: Completely incorrect or irrelevant
- 2-4: Mostly incorrect but somewhat related
- 4-6: Partially correct but missing key elements
- 6-8: Mostly correct with minor errors or omissions
- 8-10: Fully correct and matches ground-truth perfectly

Our evaluation will only take into account the first number in your response, so make sure it begins with the score (no additional text, no quotation marks, no spaces, no new lines, ...).
'''


prompt_from_hf_judge_base_prompt_v2 =  lambda x: f'''I require an evaluation of a model's generated answer against the ground-truth answer, considering how well it addresses the question and utilizes the context. I'll provide all necessary elements for comparison.

## Input Data

{{
    "context": """{x['context']}""",
    "question": """{x['question']}""",
    "ground_truth": """{x['target']}""",
    "generated_answer": """{x['output']}"""
}}

## Evaluation Criteria

1. **Ground Truth Match**: How accurately the generated answer matches the ground truth (primary factor)
2. **Question Relevance**: Whether the answer properly addresses the specific question asked
3. **Context Utilization**: How effectively the answer incorporates relevant context

## Scoring Guide

Provide a single score from 0-10 based on:
- 0-2: Fails on all criteria (incorrect facts, irrelevant to question, ignores context)
- 3-4: Poor performance (mostly incorrect but somewhat related to question/context)
- 5-6: Mixed quality (partially correct but with notable issues in question focus or context use)
- 7-8: Good (mostly correct with minor question/context mismatches)
- 9-10: Excellent (fully correct, perfectly addresses question, optimally uses context)

Our evaluation will only take into account the first number in your response, so make sure it begins with the score (no additional text, no quotation marks, no spaces, no new lines, ...).
'''


class JudgeCompare:
    def __init__(self, dataset, model_name, max_new_tokens, seed=42,
                 use_vllm=False, use_api=False,
                 model_url=None, tokenizer_name=None,
                 judge_prompt=None, use_proxy=None):
        self.dataset = dataset
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        set_seed(self.seed)
        self.use_vllm = use_vllm
        self.use_api = use_api
        if self.use_api:
            self.model_url = model_url
            self.tokenizer_name = tokenizer_name
        self.use_proxy = use_proxy
        self.init_model_tokenizer()
        if judge_prompt is None:
            self.judge_prompt = instruction_score_only_lambda
        else:
            self.judge_prompt = judge_prompt

    def init_model_tokenizer(self):
        if self.use_vllm:
            self.init_model_tokenizer_vllm()
        elif self.use_api:
            self.init_model_tokenizer_api()
        else:
            self.init_model_tokenizer_hf()

    def init_model_tokenizer_hf(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="auto")
        self.generation_config = GenerationConfig.from_pretrained(
            self.model_name, temperature=1.0, top_p=1.0, do_sample=False,
            max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
        )

    def init_model_tokenizer_vllm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       trust_remote_code=True)
        self.model = LLM(self.model_name,
                         tokenizer=self.model_name,
                         tensor_parallel_size=torch.cuda.device_count(),
                         trust_remote_code=True,
                         dtype=torch.bfloat16,
                         seed=self.seed)
        self.generation_params = SamplingParams(max_tokens=self.max_new_tokens,
                                                temperature=1.0,
                                                top_p=1.0, detokenize=True,
                                                seed=self.seed)

    def init_model_tokenizer_api(self):
        # for this case, one should separately set tokenizer and model
        # tokenizer could be not the same
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,
                                                       trust_remote_code=True)
        if self.use_proxy is not None:
            self.model = OpenAI(http_client=httpx.Client(proxy=self.use_proxy),
                                api_key=os.environ["API_KEY"].encode('ascii', 'ignore').decode('ascii'), base_url=self.model_url)
        else:
            self.model = OpenAI(api_key=os.environ["API_KEY"], base_url=self.model_url)
        self.generation_params = {
            "max_completion_tokens": self.max_new_tokens,
            "temperature": 1.0,
            "top_p": 1.0
        }

    def compare_answers(self, example):
        if self.use_vllm:
            return self.compare_answers_vllm(example)
        elif self.use_api:
            return self.compare_answers_api(example)
        else:
            return self.compare_answers_hf(example)

    def compare_answers_api(self, message):
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message,
            stream=False,
            **self.generation_params,
        )
        judge_answer = response.choices[0].message.content
        return judge_answer

    def compare_answers_vllm(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                       add_generation_prompt=True, 
                                                       tokenize=True,
                                                       return_dict=False)
        outputs = self.model.generate(
            TokensPrompt({"prompt_token_ids": input_ids}),
            sampling_params=self.generation_params,
            use_tqdm=False,
        )
        curr_response = outputs[0].outputs[0].text
        return curr_response

    def parse_answer(self, response):
        # split answer and extract similarity score
        try:
            score = response.split("Similarity Score:")[1]
            if "**" in score:
                score = score.split("**")[0]
            if "/" in score:
                score = score.split("/")[0].strip()
        except:
            score = response
        return score

    def build_message(self, el):
        message=[
                #{"role": "system", "content": "You are a helpful assistant. You should compare two answers on the given question with respect to context, and state are the answers similar or not"},
                {"role": "user", "content": self.judge_prompt(el)},
        ]
        return message

    def save_to_csv(self, save_path, data):
        df = pd.DataFrame.from_dict(data)
        df.to_csv(save_path, index=False)

    def process_sequences(self, save_during_process=0, save_path=None):
        # calc task lengths
        all_responses = {
            "target": [],
            "output": [],
            "question": [],
            "context": [],
            "responses": [],
            "scores": [],
        }
        for idx, el in tqdm(self.dataset.iterrows()):
            buf_dict = dict(el)
            for key in list(all_responses.keys())[:4]:
                all_responses[key].append(buf_dict[key])
            message = self.build_message(buf_dict)
            judge_response = self.compare_answers(message)
            judge_score = self.parse_answer(judge_response)
            all_responses["responses"].append(judge_response)
            all_responses["scores"].append(judge_score)
            if save_during_process > 0:
                if idx % save_during_process == 0 and idx != 0:
                    self.save_to_csv(save_path, all_responses)
        self.save_to_csv(save_path, all_responses)
        return all_responses

def main(args):
    input_dataset = pd.read_csv(args.input_path)
    if args.judge_prompt is None:
        judge_prompt = instruction_score_only_lambda
    elif args.judge_prompt == "hf_mod":
        judge_prompt = prompt_from_hf_judge_base_prompt_v2
    judge = JudgeCompare(dataset=input_dataset,
                         model_name=args.model_name,
                         max_new_tokens=args.max_new_tokens, 
                         seed=args.seed,
                         use_vllm=args.use_vllm,
                         use_api=args.use_api,
                         tokenizer_name=args.tokenizer_name,
                         model_url=args.model_url,
                         judge_prompt=judge_prompt,
                         use_proxy=args.use_proxy)
    if args.save_path is not None:
        save_file = os.path.join(args.save_path, "judge_res.csv")
    else:
        file_suffix = "_judge_res.csv"
        if args.judge_prompt == "hf_mod":
            file_suffix = "_judge_res_hf_mod.csv"
            if args.use_api:
                file_suffix = f"_judge_res_api_{args.model_name}_hf_mod.csv"
        elif args.use_api:
            file_suffix = f"_judge_res_api_{args.model_name}.csv"
        if args.model_name != "meta-llama/Llama-3.1-8B-Instruct":
            save_file = args.input_path[:-4] + f"_{args.model_name.split('/')[-1]}{file_suffix}"
        else:
            save_file = args.input_path[:-4] + file_suffix
    print(save_file)
    all_responses = judge.process_sequences(save_during_process=args.save_steps,
                                            save_path=save_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--input_path', type=str, default="../../data/long_context_raw/pile_unc", help='input dataset for labelling')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='model name')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new tokens for generation')
    parser.add_argument('--save_steps', type=int, default=100, help='Save dataset during labelling each save_steps')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--use_vllm', action='store_true', default=False, help="Use VLLM for generation")
    parser.add_argument('--use_api', action='store_true', default=False, help="Use OpenAI api for generation")
    parser.add_argument('--model_url', type=str, default="https://api.deepseek.com", help='URL for api')
    parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='tokenizer name')
    parser.add_argument('--judge_prompt', type=str, default=None, help='prompt for judge')
    parser.add_argument('--use_proxy', type=str, default=None, help='proxy for api')
    args = parser.parse_args()
    main(args)
