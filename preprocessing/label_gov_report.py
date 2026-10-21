import numpy as np
import os
import copy
import torch
import datasets
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, GenerationConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt, TextPrompt
from openai import OpenAI


long_context_tasks = [
    """Given the above text, generate 5 diverse general questions and answers that require a comprehensive understanding of the text in detail. 
    Please provide the question and answer in the following format: 
    '1: Question: \n Answer \n ', '2: Question: \n Answer \n ',...""",
    """Given the above text, generate 5 diverse questions and answers that require summarizing several parts of the text. 
    Please provide the question and answer in the following format: 
    '1: Question: \n Answer \n ', '2: Question: \n Answer \n ',...""",
    """Given the above text, generate 5 diverse questions and answers that require multi-hop reasoning on different parts of the text. 
    Please provide the question and answer in the following format: 
    '1: Question: \n Answer \n ', '2: Question: \n Answer \n ',...""",
    """Given the above text, generate 5 diverse questions and answers that require extracting information from different parts of the text. 
    Please provide the question and answer in the following format: 
    '1: Question: \n Answer \n ', '2: Question: \n Answer \n ',...""",
    """Given the above text, generate 5 diverse questions and answers that require question to be answered using several parts of the text. 
    Please provide the question and answer in the following format: 
    '1: Question: \n Answer \n ', '2: Question: \n Answer \n ',...""",
]


long_context_tasks_cot = [
    """Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material. 

For each question, construct a detailed answer that includes:

1. Restatement of the original question to ensure clarity.
2. A logical chain of thought that demonstrates critical engagement with the text, offering insights and interpretations.
3. A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Present each question and answer pair in the following format:

'1: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Develop a thoughtful analysis leading to the conclusion. Provide a concise and complete final answer.]

'2: Question:  
[Your question here]

Answer:  
[Follow the same structured response as above.]

...

Continue this format for all five questions and answers. Ensure that each question explores a different aspect of the text to showcase a broad understanding.""",
    """Given the above text, generate 5 diverse questions and answers that require summarizing several parts of the text. 
    
For each question, construct a detailed answer that includes:

1. Restatement of the original question to ensure clarity.
2. A logical chain of thought that demonstrates critical engagement with the text, offering insights and interpretations.
3. A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Present each question and answer pair in the following format:

'1: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Develop a thoughtful analysis leading to the conclusion. Provide a concise and complete final answer.]

'2: Question:  
[Your question here]

Answer:  
[Follow the same structured response as above.]

...

Continue this format for all five questions and answers. Ensure that each question explores a different aspect of the text to showcase a broad understanding.""",
    """Given the above text, generate 5 diverse questions and answers that require multi-hop reasoning on different parts of the text. 
    
For each question, construct a detailed answer that includes:

1. Restatement of the original question to ensure clarity.
2. A logical chain of thought that demonstrates critical engagement with the text, offering insights and interpretations.
3. A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Present each question and answer pair in the following format:

'1: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Develop a thoughtful analysis leading to the conclusion. Provide a concise and complete final answer.]

'2: Question:  
[Your question here]

Answer:  
[Follow the same structured response as above.]

...

Continue this format for all five questions and answers. Ensure that each question explores a different aspect of the text to showcase a broad understanding.""",
    """Given the above text, generate 5 diverse questions and answers that require extracting information from different parts of the text. 
    
For each question, construct a detailed answer that includes:

1. Restatement of the original question to ensure clarity.
2. A logical chain of thought that demonstrates critical engagement with the text, offering insights and interpretations.
3. A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Present each question and answer pair in the following format:

'1: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Develop a thoughtful analysis leading to the conclusion. Provide a concise and complete final answer.]

'2: Question:  
[Your question here]

Answer:  
[Follow the same structured response as above.]

...

Continue this format for all five questions and answers. Ensure that each question explores a different aspect of the text to showcase a broad understanding.""",
    """Given the above text, generate 5 diverse questions and answers that require question to be answered using several parts of the text. 
    
For each question, construct a detailed answer that includes:

1. Restatement of the original question to ensure clarity.
2. A logical chain of thought that demonstrates critical engagement with the text, offering insights and interpretations.
3. A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Present each question and answer pair in the following format:

'1: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Develop a thoughtful analysis leading to the conclusion. Provide a concise and complete final answer.]

'2: Question:  
[Your question here]

Answer:  
[Follow the same structured response as above.]

...

Continue this format for all five questions and answers. Ensure that each question explores a different aspect of the text to showcase a broad understanding.""",
]


common_qa_instruction = """

For each question, construct a detailed answer that includes:

1. Repeating and rephrasing the original question to ensure clarity.
2. A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Present each question and answer pair in the following format:

'1: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Provide a concise and complete final answer.]

'2: Question:  
[Your question here]

Answer:  
[Follow the same structured response as above.]

...

Continue this format for all five questions and answers. Ensure that each question explores a different aspect of the text to showcase a broad understanding."""


common_qa_inst_instruction = """

For each question, provide:

1. The question – Ensure it is thought-provoking and requires an in-depth response.  
2. A structured answer that includes:  
    - Repeating and rephrasing the original question to ensure clarity.
    - A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.
3. A generated instruction that explains the task being performed (e.g., answering a question, summarizing, analyzing, comparing) and specifies how the answer should be structured. The instruction should be derived from the nature of the task and should be suitable for someone attempting a similar question.  

Each question-answer-instruction set should be presented in the following structure:

'1: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Provide a concise and complete final answer.]

Instruction:  
[State what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]  

'2: Question:  
[Your question here]

Answer:  
[Restate the question to set context. Provide a concise and complete final answer.]

Instruction:  
[State what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]

...

Repeat the format for five question-answer-instruction sets, ensuring each explores a different aspect of the text."""



common_qa_inst_instruction_no_repeat = """

For each question, provide:

1. The question – Ensure it is thought-provoking and requires an in-depth response.  
2. A structured answer that includes:  
    - A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.
3. A generated instruction that explains the task being performed (e.g., answering a question, summarizing, analyzing, comparing) and specifies how the answer should be structured. The instruction should be derived from the nature of the task and should be suitable for someone attempting a similar question.  

Each question-answer-instruction set should be presented in the following structure:

'1: Question:  
[Your question here]

Answer:  
[Provide a concise and complete final answer.]

Instruction:  
[State what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]  

'2: Question:  
[Your question here]

Answer:  
[Provide a concise and complete final answer.]

Instruction:  
[State what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]

...

Repeat the format for five question-answer-instruction sets, ensuring each explores a different aspect of the text."""


common_qa_only_inst = """

For each question, provide:

1. The question with detailed instruction – Ensure question is thought-provoking and requires an in-depth response, and instruction explains the task being performed (e.g., answering a question, summarizing, analyzing, comparing) and specifies how the answer should be structured. The instruction should be derived from the nature of the task and should be suitable for someone attempting a similar question.
2. A structured answer that includes:  
    - Repeating the original question with instruction to ensure clarity.
    - A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Each question-answer set should be presented in the following structure:

'1: Question:  
[Your question here. Also state what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]

Answer:  
[Repeat the question and instruction to set context. Provide a concise and complete final answer.]

'2: Question:  
[Your question here. Also state what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]

Answer:  
[Repeat the question to set context. Provide a concise and complete final answer.]

...

Repeat the format for five question-answer sets, ensuring each explores a different aspect of the text."""


common_qa_only_inst_no_repeat = """

For each question, provide:

1. The question with detailed instruction – Ensure question is thought-provoking and requires an in-depth response, and instruction explains the task being performed (e.g., answering a question, summarizing, analyzing, comparing) and specifies how the answer should be structured. The instruction should be derived from the nature of the task and should be suitable for someone attempting a similar question.
2. A structured answer that includes:  
    - A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Each question-answer set should be presented in the following structure:

'1: Question:  
[Your question here. Also state what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]

Answer:  
[Provide a concise and complete final answer.]

'2: Question:  
[Your question here. Also state what type of task is being performed (in general format - without text-specific details, e.g., answering a question, summarizing, explaining, analyzing, etc.). Describe how the answer should be structured. This may include restating the question, breaking down key points, providing textual evidence, making logical connections, and concluding with a direct response.]

Answer:  
[Provide a concise and complete final answer.]

...

Repeat the format for five question-answer sets, ensuring each explores a different aspect of the text."""


long_context_tasks_q_in_a = [
    "Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material." + common_qa_instruction,
    "Given the above text, generate five diverse questions and answers that require summarizing several parts of the text." + common_qa_instruction,
    "Given the above text, generate five diverse questions and answers that require multi-hop reasoning on different parts of the text." + common_qa_instruction, 
    "Given the above text, generate five diverse questions and answers that require extracting information from different parts of the text." + common_qa_instruction,
    "Given the above text, generate five diverse questions and answers that require question to be answered using several parts of the text." + common_qa_instruction,
]

long_context_tasks_q_in_a_inst = [
    "Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that require summarizing several parts of the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that require multi-hop reasoning on different parts of the text." + common_qa_inst_instruction, 
    "Given the above text, generate five diverse questions that require extracting information from different parts of the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that require question to be answered using several parts of the text." + common_qa_inst_instruction,
]

long_context_tasks_q_not_in_a_inst = [
    "Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that require summarizing several parts of the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that require multi-hop reasoning on different parts of the text." + common_qa_inst_instruction_no_repeat, 
    "Given the above text, generate five diverse questions that require extracting information from different parts of the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that require question to be answered using several parts of the text." + common_qa_inst_instruction_no_repeat,
]

long_context_tasks_q_in_a_inst_only_50 = [
    "Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that require summarizing several parts of the text." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that require multi-hop reasoning on different parts of the text." + common_qa_only_inst, 
    "Given the above text, generate five diverse questions that require extracting information from different parts of the text." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that require question to be answered using several parts of the text." + common_qa_only_inst,
    "Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that require summarizing several parts of the text." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that require multi-hop reasoning on different parts of the text." + common_qa_only_inst_no_repeat, 
    "Given the above text, generate five diverse questions that require extracting information from different parts of the text." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that require question to be answered using several parts of the text." + common_qa_only_inst_no_repeat,
]


easy_long_context_tasks_q_in_a_inst_only_50 = [
    "Given the above text, generate five diverse questions that require summarizing the main points of the text in 2-3 sentences each." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that ask for the identification of the most important keywords or phrases in the text." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that require locating specific facts or details within the text." + common_qa_only_inst, 
    "Given the above text, generate five diverse questions that ask for rephrasing specific sections of the text in simpler terms." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that ask for identifying the locations, settings, or contexts described in the text." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that ask for specific examples or instances mentioned in the text to support a concept or claim." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that ask for the definition or explanation of specific terms or concepts mentioned in the text." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that ask for lists of items, such as examples, steps, or components, described in the text." + common_qa_only_inst,
    "Given the above text, generate five diverse questions that ask whether specific statements or facts are true or false based on the information in the text." + common_qa_only_inst,

    "Given the above text, generate five diverse questions that require summarizing the main points of the text in 2-3 sentences each." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that ask for the identification of the most important keywords or phrases in the text." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that require locating specific facts or details within the text." + common_qa_only_inst_no_repeat, 
    "Given the above text, generate five diverse questions that ask for rephrasing specific sections of the text in simpler terms." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that ask for identifying the locations, settings, or contexts described in the text." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that ask for specific examples or instances mentioned in the text to support a concept or claim." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that ask for the definition or explanation of specific terms or concepts mentioned in the text." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that ask for lists of items, such as examples, steps, or components, described in the text." + common_qa_only_inst_no_repeat,
    "Given the above text, generate five diverse questions that ask whether specific statements or facts are true or false based on the information in the text." + common_qa_only_inst_no_repeat,
]



long_context_tasks_qa_inst_only_50 = [
    "Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that require summarizing several parts of the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that require multi-hop reasoning on different parts of the text." + common_qa_inst_instruction, 
    "Given the above text, generate five diverse questions that require extracting information from different parts of the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that require question to be answered using several parts of the text." + common_qa_inst_instruction,
    "Given the provided text, generate five diverse and thought-provoking questions that necessitate a deep and comprehensive understanding of the material." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that require summarizing several parts of the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that require multi-hop reasoning on different parts of the text." + common_qa_inst_instruction_no_repeat, 
    "Given the above text, generate five diverse questions that require extracting information from different parts of the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that require question to be answered using several parts of the text." + common_qa_inst_instruction_no_repeat,
]


easy_long_context_tasks_qa_inst_only_50 = [
    "Given the above text, generate five diverse questions that require summarizing the main points of the text in 2-3 sentences each." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that ask for the identification of the most important keywords or phrases in the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that require locating specific facts or details within the text." + common_qa_inst_instruction, 
    "Given the above text, generate five diverse questions that ask for rephrasing specific sections of the text in simpler terms." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that ask for identifying the locations, settings, or contexts described in the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that ask for specific examples or instances mentioned in the text to support a concept or claim." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that ask for the definition or explanation of specific terms or concepts mentioned in the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that ask for lists of items, such as examples, steps, or components, described in the text." + common_qa_inst_instruction,
    "Given the above text, generate five diverse questions that ask whether specific statements or facts are true or false based on the information in the text." + common_qa_inst_instruction,

    "Given the above text, generate five diverse questions that require summarizing the main points of the text in 2-3 sentences each." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that ask for the identification of the most important keywords or phrases in the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that require locating specific facts or details within the text." + common_qa_inst_instruction_no_repeat, 
    "Given the above text, generate five diverse questions that ask for rephrasing specific sections of the text in simpler terms." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that ask for identifying the locations, settings, or contexts described in the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that ask for specific examples or instances mentioned in the text to support a concept or claim." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that ask for the definition or explanation of specific terms or concepts mentioned in the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that ask for lists of items, such as examples, steps, or components, described in the text." + common_qa_inst_instruction_no_repeat,
    "Given the above text, generate five diverse questions that ask whether specific statements or facts are true or false based on the information in the text." + common_qa_inst_instruction_no_repeat,
]

test_sys_prompt = """
"You are an advanced language model tasked with generating high-quality, diverse, and realistic synthetic data for instruction tuning. Your goal is to create examples that mimic human-like interactions, covering a wide range of tasks, domains, and complexities. Each example should consist of:

1. **Instruction**: A clear, concise, and specific task description that a user might provide. Ensure the instruction is unambiguous and actionable.
2. **Input**: Optional context or input data required to complete the task. This could be a question, a paragraph, a table, or any relevant information.
3. **Output**: A well-structured, accurate, and contextually appropriate response to the instruction, based on the input (if provided).

**Guidelines for Data Generation:**
- **Diversity**: Cover a broad spectrum of tasks, including but not limited to text summarization, question answering, code generation, translation, reasoning, creative writing, and problem-solving.
- **Complexity**: Vary the difficulty level of tasks, from simple to complex, to ensure the model can handle a wide range of scenarios.
- **Realism**: Ensure the examples reflect real-world use cases and are free from biases, stereotypes, or harmful content.
- **Consistency**: Maintain logical coherence between the instruction, input (if any), and output.
- **Creativity**: Introduce novel and imaginative scenarios where appropriate, while keeping the examples plausible and useful for instruction tuning.

**Example Format:**
- **Instruction**: "Summarize the following paragraph in one sentence."
- **Input**: "The rapid advancement of artificial intelligence has led to significant breakthroughs in various fields, including healthcare, finance, and education. However, it also raises ethical concerns about privacy, bias, and job displacement."
- **Output**: "AI advancements bring breakthroughs in healthcare, finance, and education but raise ethical concerns about privacy, bias, and job displacement."

**Note**: Avoid generating repetitive or overly simplistic examples. Prioritize quality, relevance, and utility for training language models to follow instructions effectively."

"""

simple_sys_prompt = """
"You are an advanced language model tasked with generating high-quality, diverse, and realistic synthetic data for instruction tuning. Your goal is to create examples that mimic human-like interactions, covering a wide range of tasks, domains, and complexities.
**Guidelines for Data Generation:**
- **Diversity**: Cover a broad spectrum of tasks, including but not limited to text summarization, question answering, code generation, translation, reasoning, creative writing, and problem-solving.
- **Complexity**: Vary the difficulty level of tasks, from simple to complex, to ensure the model can handle a wide range of scenarios.
- **Realism**: Ensure the examples reflect real-world use cases and are free from biases, stereotypes, or harmful content.
- **Consistency**: Maintain logical coherence between the instruction, input (if any), and output.
- **Creativity**: Introduce novel and imaginative scenarios where appropriate, while keeping the examples plausible and useful for instruction tuning.
**Note**: Avoid generating repetitive or overly simplistic examples. Prioritize quality, relevance, and utility for training language models to follow instructions effectively."
"""
simple_sys_prompt_template = lambda context, task_text: [
    {"role": "system", "content": simple_sys_prompt},
    {"role": "user", "content": f"*****Start of the reference text:*****\n {context}\n *****End of the reference text:*****\n {task_text}"},
]




gov_report_sys_prompt = """
You are an expert NLP researcher specializing in information extraction and question answering. Your task is to generate high-quality question-answer (QA) pairs from government reports in the GovReport dataset. Each QA pair should require a detailed understanding of the given text, avoiding trivial or overly general questions.

Guidelines for QA Generation:
    Complexity & Depth:
        Prefer multi-faceted questions that require synthesis of information.
        Avoid yes/no or overly simplistic questions (e.g., "What is the report about?").
        Favor questions that need reasoning, comparisons, or explanations.
    Question Types:
        Fact-based: "According to the report, what are the three key challenges in implementing X policy?"
        Analytical: "How does the report justify the proposed increase in funding for Y program?"
        Comparative: "What differences does the report highlight between Method A and Method B?"
        Cause/Effect: "What factors does the report identify as contributing to the decline in Z metric?"
    Answer Requirements:
        Answers must be directly grounded in the provided text.
        Include specific details, numbers, or quotes where applicable.
        If a question cannot be answered from the text alone, discard it.
        Answer should be concise, complete and not longer than three sentences.
"""

gov_report_qa_only = """

For each question, provide:

1. The question – Ensure question is thought-provoking and requires an in-depth response, and instruction explains the task being performed (e.g., answering a question, summarizing, analyzing, comparing) and specifies how the answer should be structured.
2. A structured answer that includes:
    - A conclusive and well-articulated response that directly addresses the question, drawing from the text's details.

Each question-answer set should be presented in the following structure:

'1: Question:  
[Your question here.]

Answer:  
[Provide a concise and complete final answer.]

'2: Question:  
[Your question here.]

Answer:  
[Provide a concise and complete final answer.]

...

Repeat the format for five question-answer sets, ensuring each explores a different aspect of the text."""

gov_report_tasks = [
    """Based on the given text, generate 5 high-quality QA pairs that require a detailed understanding of the content. Ensure:
    Questions are non-trivial and require careful reading.
    Answers are precise and directly supported by the text.
    Avoid redundancy (each question should focus on a distinct aspect).\n""" + gov_report_qa_only,
]


gov_report_prompt_template = lambda context, task_text: [
    {"role": "system", "content": gov_report_sys_prompt},
    {"role": "user", "content": f"*****Start of the reference text:*****\n {context}\n *****End of the reference text:*****\n {task_text}"},
]

gov_report_prompt_template_no_sys = lambda context, task_text: [
    {"role": "user", "content": f"*****Start of the reference text:*****\n {context}\n *****End of the reference text:*****\n {gov_report_sys_prompt + task_text}"},
]


#    The answer should contain a recollection of the original question, chain of thoughts leading to the final answer, and the full answer to the question in the concluding sentence. 
#    The answer should contain a recollection of the original question, step-by-step thoughts leading to the final answer, and the answer itself. 
class LongContextLabeller:
    def __init__(self, dataset, text_field, model_name, max_new_tokens, segment_size, seed=42,
                 use_vllm=False, template=None, aggregation_template=None, use_api=False,
                 model_url=None, tokenizer_name=None, temperature=1.0, topp=1.0):
        self.dataset = dataset
        self.text_field = text_field
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.segment_size = segment_size
        self.seed = seed
        set_seed(self.seed)
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.temperature = temperature
        self.topp = topp
        if self.use_api:
            self.model_url = model_url
            self.tokenizer_name = tokenizer_name
        if template is None:
            self.template = lambda context, task_text: [
                {"role": "user", "content": f"*****Start of the reference text:*****\n {context}\n *****End of the reference text:*****\n {task_text}"},
            ]
        else:
            self.template = template
        if aggregation_template is None:
            self.aggregation_template = lambda text: [
                {"role": "user", "content": f"*****Question-answer pairs*****\n {text}\n *****End of question-answer pairs*****\n Given the above question-answer pairs, generate one question-answer pair which combine information from all given pairs. Please provide question and answer in the following format: 'Question: \n Answer \n '"},
            ]
        else:
            self.aggregation_template = aggregation_template
        self.init_model_tokenizer()

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
            self.model_name, temperature=self.temperature, top_p=self.topp, do_sample=False,
            max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
        )

    def split_response_by_qa(self, text):
        qa_pairs = []
        idx = 1
        while idx <= 5:
            # sometimes we have less than 5 questions
            if text.find(f"{idx}: Question") != -1:
                qa_pair = "Question" + text.split(f"{idx}: Question")[1]
                if text.find(f"{idx+1}: Question") != -1:
                    qa_pair = qa_pair.split(f"{idx+1}: Question")[0]
            else:
                # firstly try another splitter
                if text.find(f"{idx}: ") != -1:
                    qa_pair = "Question: " + text.split(f"{idx}: ")[1]
                    if text.find(f"{idx+1}: ") != -1:
                        qa_pair = qa_pair.split(f"{idx+1}: ")[0]
                else:
                    print(f"Only {idx-1} QA pairs were generated")
                    #print(text)
                    break
            qa_pairs.append(qa_pair)
            idx += 1
        return qa_pairs

    def generate_with_pars(self, messages):
        if self.use_vllm:
            return self.generate_with_pars_vllm(messages)
        elif self.use_api:
            return self.generate_with_pars_api(messages)
        else:
            return self.generate_with_pars_hf(messages)

    def generate_with_pars_hf(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                       add_generation_prompt=True, 
                                                       return_tensors="pt",
                                                       return_dict=False).to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            generation_config=self.generation_config,
        )
        curr_response = outputs[0][input_ids.shape[-1]:]
        curr_text_response = self.tokenizer.decode(curr_response, skip_special_tokens=True)
        return curr_text_response

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
                                                temperature=self.temperature,
                                                top_p=self.topp, detokenize=True,
                                                seed=self.seed)

    def generate_with_pars_vllm(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                       add_generation_prompt=True, 
                                                       tokenize=True,
                                                       return_dict=False,
                                                       truncation=True,
                                                       enable_thinking=False)
        try:
            outputs = self.model.generate(
                TokensPrompt({"prompt_token_ids": input_ids}),
                sampling_params=self.generation_params,
                use_tqdm=False,
            )
            curr_response = outputs[0].outputs[0].text
        except:
            curr_response = ""
        return curr_response

    def init_model_tokenizer_api(self):
        # for this case, one should separately set tokenizer and model
        # tokenizer could be not the same
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,
                                                       trust_remote_code=True)
        self.model = OpenAI(api_key=os.environ["API_KEY"], base_url=self.model_url)
        self.generation_params = {
            "model": self.model_name,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.topp
        }

    def generate_with_pars_api(self, messages):
        buf_generation = copy.deepcopy(self.generation_params)
        buf_generation["messages"] = messages
        outputs = self.model.chat.completions.create(
            **buf_generation
        )
        curr_response = outputs.choices[0].message.content
        return curr_response

    def aggregate_responses(self, responses):
        aggregated_response = ""
        splitted_responses = [self.split_response_by_qa(el) for el in responses]
        min_n_answers = min([len(el) for el in splitted_responses])
        # workaround to not take into account non-parsed responses
        if min_n_answers == 0:
            new_splitted_responses = [el for el in splitted_responses if len(el) != 0]
            if len(new_splitted_responses) != 0:
                splitted_responses = new_splitted_responses
            min_n_answers = min([len(el) for el in splitted_responses])
        for answer_idx in range(min_n_answers):
            curr_responses = "\n".join([el[answer_idx] for el in splitted_responses])
            messages = self.aggregation_template(curr_responses)
            curr_text_response = self.generate_with_pars(messages)
            aggregated_response += f"{answer_idx+1}: {curr_text_response}\n\n"
        return aggregated_response

    def split_full_context(self, full_context, task_text, task_len=None):
        # firstly calc length of tokenized task_text+template
        if task_len is None:
            messages = self.template("", task_text)
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            task_len = len(input_ids[0])
        real_segment_size = self.segment_size - task_len
        contexts = []
        input_ids = self.tokenizer(full_context, return_tensors="pt")["input_ids"][0]
        segm_nums = len(input_ids) // real_segment_size + int(len(input_ids) % real_segment_size != 0)
        for segm in range(segm_nums):
            contexts.append(self.tokenizer.decode(input_ids[segm*real_segment_size: (segm+1)*real_segment_size], skip_special_tokens=True))
        return contexts

    def process_short_sentence(self, sample, task_text):
        context = sample[self.text_field]
        messages = self.template(context, task_text)
        text_response = self.generate_with_pars(messages)
        # TODO: check the last qa pair for the generation end
        return {"context": context, "task": task_text, "response": text_response, "aggregated": False, "all_responses": []}

    def process_long_sentence(self, sample, task_text, task_len=None):
        # process sentences longer than specified segment_size
        full_context = sample[self.text_field]
        # split context on chunks of segment_size-len(task_text) - TODO
        contexts = self.split_full_context(full_context, task_text, task_len)
        responses = []
        #print(len(contexts))
        for context in contexts:
            messages = self.template(context, task_text)
            text_response = self.generate_with_pars(messages)
            responses.append(text_response)
            #print(context)
            #print(text_response)
        # generate aggregated response
        aggregated_response = self.aggregate_responses(responses)
        return {"context": full_context, "task": task_text, "response": aggregated_response, "aggregated": True, "all_responses": responses}

    def postprocess_response(self, response, quest_in_ans, gen_inst=False):
        # postprocess generated response - split qa pairs, chooose one pair randomly
        qa_pairs = self.split_response_by_qa(response)
        # filter pairs with qa
        qa = {}
        qa_pairs = [qa for qa in qa_pairs if "Answer" in qa]
        if len(qa_pairs) == 0:
            qa["question"] = ""
            qa["answer"] = ""
            qa["instruction"] = ""
            return qa
        if not(quest_in_ans):
            qa_pair = np.random.choice(qa_pairs, 1)[0]
        else:
            # adjust probability to skip the last qa pair, cause its sometimes broken
            ids = list(range(max(len(qa_pairs)-1, 1)))
            qa_pair = qa_pairs[np.random.choice(ids, 1)[0]]
        try:
            question, answer = qa_pair.split("Question: ", maxsplit=1)[1].split("Answer: ", maxsplit=1)
            if gen_inst:
                answer, instruction = answer.split("Instruction: ", maxsplit=1)
        except:
            try:
                question, answer = qa_pair.split("Question", maxsplit=1)[1].split("Answer", maxsplit=1)
                if gen_inst:
                    answer, instruction = answer.split("Instruction", maxsplit=1)
            except:
                print("Unable to parse QA pair!")
                qa["question"] = ""
                qa["answer"] = ""
                qa["instruction"] = ""
                return qa
        if not(quest_in_ans):
            qa["question"] = question.strip().rsplit("\n", 1)[0]
            qa["answer"] = answer.strip().rsplit("\n\n", 1)[0]
            if gen_inst:
                qa["instruction"] = instruction.strip().rsplit("\n\n", 1)[0]
        else:
            qa["question"] = question.strip(" \n:'").strip("[]")
            qa["answer"] = answer.strip(" \n:'").strip("[]")
            if gen_inst:
                qa["instruction"] = instruction.strip(" \n:'").strip("[]")
        return qa

    def process_sequences(self, tasks, samples_num, save_during_process=0, save_path=None, quest_in_ans=False, gen_inst=False):
        # randomly choose sample from dataset train
        new_data_dict = {
            "doc_id": [],
            "document_sections": [],
            "question_summary_pairs": [],
            "document": [],
            "golden_pars": [],
        }
        base_keys = ["doc_id", "document_sections", "question_summary_pairs", "document"]
        for idx in tqdm(range(samples_num)):
            sample_idx = np.random.randint(0, len(self.dataset["train"]))
            # randomly sample from 1 to 5 paragraphs from chosen sample
            num_of_pars = np.random.randint(1, 6)
            par_ids_list = np.arange(len(self.dataset["train"][sample_idx]["document_sections"]["paragraphs"]))
            par_ids = np.random.choice(par_ids_list, min(num_of_pars, len(par_ids_list)), replace=False)
            current_text = [self.dataset["train"][sample_idx]["document_sections"]["title"][par_idx] + "\n" + self.dataset["train"][sample_idx]["document_sections"]["paragraphs"][par_idx] for par_idx in par_ids]
            current_text = '\n'.join(current_text).strip("\n")
            # TODO: generate QA pair based on current text
            buf_response = self.process_short_sentence({self.text_field: current_text}, gov_report_tasks[0])
            # TODO: parse QA pair
            qa = self.postprocess_response(buf_response["response"], quest_in_ans, gen_inst)
            #print(qa)
            # TODO: copy sample_idx sample and add to buf_dataset
            # TODO: check qa, look at data format, save to new field
            if len(qa["question"]) > 0:
                for base_key in base_keys:
                    new_data_dict[base_key].append(self.dataset["train"][sample_idx][base_key])
                new_golden_pars = {
                    "paragraphs": [[self.dataset["train"][sample_idx]["document_sections"]["paragraphs"][par_idx] for par_idx in par_ids]],
                    "paragraphs_ids": [par_ids],
                    "questions": [[qa["question"]]],
                    "summaries": [[qa["answer"]]],
                    "titles": [[self.dataset["train"][sample_idx]["document_sections"]["title"][par_idx] for par_idx in par_ids]],
                }
                new_data_dict["golden_pars"].append(new_golden_pars)
            if save_during_process > 0:
                if idx % save_during_process == 0 and idx != 0:
                    new_dataset = datasets.Dataset.from_dict(new_data_dict)
                    new_dataset.save_to_disk(save_path)
        new_dataset = datasets.Dataset.from_dict(new_data_dict)
        new_dataset.save_to_disk(save_path)
        print(new_dataset)

    def save_to_dataset(self, save_path, responses):
        new_dataset = datasets.Dataset.from_list(responses)
        new_dataset.save_to_disk(save_path)


def main(args):
    input_dataset = datasets.load_from_disk(args.input_dataset)
    template = gov_report_prompt_template
    if "gemma" in args.model_name:
        template = gov_report_prompt_template_no_sys
    lcl = LongContextLabeller(dataset=input_dataset,
                              text_field=args.text_field,
                              model_name=args.model_name,
                              max_new_tokens=args.max_new_tokens, 
                              segment_size=args.segment_size,
                              seed=args.seed,
                              use_vllm=args.use_vllm,
                              use_api=args.use_api,
                              tokenizer_name=args.tokenizer_name,
                              model_url=args.model_url,
                              temperature=args.temperature,
                              topp=args.topp,
                              template=template)
    gen_task = long_context_tasks
    if args.add_cot_question:
        gen_task = long_context_tasks_cot
    elif args.add_quest_in_ans:
        gen_task = long_context_tasks_q_in_a_inst
        quest_in_ans = True
        gen_inst = True
    elif args.not_add_quest_in_ans:
        gen_task = long_context_tasks_q_not_in_a_inst
        quest_in_ans = True
        gen_inst = True
    elif args.partially_repeat_q:
        gen_inst = False
        if args.task_complexity == "base":
            gen_task = long_context_tasks_q_in_a_inst_only_50
        elif args.task_complexity == "easy":
            gen_task = easy_long_context_tasks_q_in_a_inst_only_50
        elif args.task_complexity == "base_inst":
            gen_task = long_context_tasks_qa_inst_only_50
            gen_inst = True
        elif args.task_complexity == "easy_inst":
            gen_task = easy_long_context_tasks_qa_inst_only_50
            gen_inst = True
        else:
            raise NotImplementedError("There is no templates for such task complexity!")
        quest_in_ans = True
    lcl.process_sequences(gen_task,
                          samples_num=args.num_samples,
                          save_during_process=args.save_steps,
                          save_path=args.save_path,
                          quest_in_ans=False,
                          gen_inst=False)
    #new_dataset = datasets.Dataset.from_list(all_responses)
    #print(new_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--input_dataset', type=str, default="../../data/long_context_raw/pile_unc", help='input dataset for labelling')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='model name')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new tokens for generation')
    parser.add_argument('--segment_size', type=int, default=8192, help='segment_size')
    parser.add_argument('--save_steps', type=int, default=100, help='Save dataset during labelling each save_steps')
    parser.add_argument('--save_path', type=str, default="../../data/long_context_raw/TEST_llama31_8b_labelled_pile_unc_processed")
    parser.add_argument('--text_field', type=str, default="context")
    parser.add_argument('--use_vllm', action='store_true', default=False, help="Use VLLM for generation")
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to label')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index of dataset')
    parser.add_argument('--add_cot_question', action='store_true', default=False, help="Use tasks and prompts for CoT and question recollection")
    parser.add_argument('--add_quest_in_ans', action='store_true', default=False, help="Use tasks and prompts for question recollection")
    parser.add_argument('--partially_repeat_q', action='store_true', default=False, help="Use tasks and prompts for question recollection, but partially and with inst in q")
    parser.add_argument('--use_api', action='store_true', default=False, help="Use OpenAI api for generation")
    parser.add_argument('--model_url', type=str, default="https://api.deepseek.com", help='URL for api')
    parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='tokenizer name')
    parser.add_argument('--task_complexity', type=str, default="base")
    parser.add_argument('--temperature', type=float, default=1.0, help='temp')
    parser.add_argument('--topp', type=float, default=1.0, help='topp')
    parser.add_argument('--use_system_prompt', action='store_true', default=False, help="Adds simple system prompt to enhance generation")
    parser.add_argument('--not_add_quest_in_ans', action='store_true', default=False, help="Use tasks and prompts without question recall, for test mostly")
    args = parser.parse_args()
    main(args)
