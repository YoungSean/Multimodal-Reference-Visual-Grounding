import json
import re

import torch
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from PIL import Image

models = {
    'internvl2_5-1b': "/data/models/huggingface-format/InternVL2_5-1B",
    'internvl2_5-2b': "/data/models/huggingface-format/InternVL2_5-2B",
    'internvl2_5-4b': "/data/models/huggingface-format/InternVL2_5-4B",
    'internvl2_5-8b': "/data/models/huggingface-format/InternVL2_5-8B",
    'qwen2.5-vl-3b': "/data/models/huggingface-format/Qwen2.5-VL-3B-Instruct",
    'qwen2.5-vl-7b': "/data/models/huggingface-format/Qwen2.5-VL-7B-Instruct",
    'llama-3.1-70b-instruct': '/data/models/huggingface-format/llama-3.1-70b-instruct',
    'llama-3.1-8b-instruct': '/data/models/huggingface-format/llama-3.1-8b-instruct',
    'qwen2.5-72b-instruct': '/data/models/huggingface-format/Qwen2.5-72B-Instruct',
    'qwen2.5-32b-instruct': '/data/models/huggingface-format/Qwen2.5-32B-Instruct',
    'qwen2.5-14b-instruct': '/data/models/huggingface-format/Qwen2.5-14B-Instruct',
    'qwen2.5-7b-instruct': '/data/models/huggingface-format/Qwen2.5-7B-Instruct',
    'qwen-vl': '/data/models/huggingface-format/Qwen-VL',
    'qwen-vl-chat': '/data/models/huggingface-format/Qwen-VL-Chat',
}

def get_model_path(model):
    return models[model]

llm = None

def init_model(model_path):
    global llm

    n_gpus = torch.cuda.device_count()

    hf_overrides = None
    if 'Qwen-VL' in model_path:
        hf_overrides = {"architectures": ["QwenVLForConditionalGeneration"]}

    # For generative models (task=generate) only
    llm = LLM(
        model=model_path,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        tensor_parallel_size=n_gpus,
        hf_overrides=hf_overrides,
    )

def call(
        message,
        schema=None,
        model='internvl2_5-1b',
        max_new_tokens=1000,
        mode='vlm',
    ):

    global llm

    model_path = models[model]
    if llm is None:
        init_model(model_path)

    # Single prompt inference
    stop_token_ids = None
    if 'internvl' == model[:8]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    guided_decoding = None
    if schema is not None:
        guided_decoding_params = GuidedDecodingParams(json=schema)

    params = SamplingParams(
        max_tokens=max_new_tokens,
        guided_decoding=guided_decoding_params,
        stop_token_ids=stop_token_ids
    )

    if 'vlm' == mode:
        outputs = llm.generate(message, params, use_tqdm=False)
    elif 'llm' == mode:
        outputs = llm.chat(message, params, use_tqdm=False)

    output_text = outputs[0].outputs[0].text.strip()
    
    # try to fix json format
    output_text = re.sub(r'\n+', '\n', output_text)
    output_text = re.sub(r' +', ' ', output_text)
    if 'qwen-vl' == model[:7]:
        output_text = re.sub(r'</s>', '', output_text)
    if '}' != output_text[-1]:
        output_text += '}'
    try:
        response_json = json.loads(output_text)
    except:
        # try to complete invalid json outputs by adding } and ]
        brackets = []
        quotes = None
        flag = True
        for i in range(len(output_text)):
            # do not handle r'''\\'''
            # ignore text braced by r" and ''
            ch = output_text[i]
            prev = '' if 0 == i else output_text[i-1]
            if quotes is not None:
                if ch == quotes and prev != '\\':
                    quotes = None
                continue

            # pair {} and []
            if ('{' == ch or '[' == ch) and prev != '\\':
                brackets.append(ch)
            elif '}' == ch and prev != '\\':
                if 0 < len(brackets) and '{' == brackets[-1]:
                    brackets.pop()
                else:
                    flag = False
                    break
            elif ']' == ch and prev != '\\':
                if 0 < len(brackets) and '[' == brackets[-1]:
                    brackets.pop()
                else:
                    flag = False
                    break
        if flag:
            if quotes is not None:
                output_text += quotes
            for b in brackets[::-1]:
                if b == '[':
                    output_text += ']'
                elif b == '{':
                    output_text += '}'

    try:
        response_json = json.loads(output_text)
    except:
        response_json = output_text
        print(response_json)
    return response_json

if '__main__' == __name__:
    from pydantic import BaseModel
    class Color(BaseModel):
        description: str
        color: str
    
    class Text(BaseModel):
        text: str
        position: str
        color: str
    
    class Extraction(BaseModel):
        shape: str
        colors: list[Color]
        texts: list[Text]
        function: str
        summary: str
    schema = Extraction.model_json_schema()

    system_prompt = (
        'You are an expert at structured data extraction. '
        'You will be given a picture. '
        'Please extract information and convert it into the given structure.'
    )

    prompt = (
        'You are given an image of an item on a flat surface (on a table, ground, etc.). '
        'Please first carefully read and understand the image in detail. '
        'If there are multiple items, only carefully look through one of them. '
        'Then, describe the item in detail by following the steps and format below.\n'
        '1. Shape: Please describe the shape or type of the item, such as a bottle, '
        'bag, round item, square item, etc.\n'
        '2. Colors: Please describe all the colors on or in the item, such as label colors, '
        'text colors, cover colors, etc. The item may be covered by multiple colors. '
        'Please describe all of them one by one. For example, bottle: transparent, liquid '
        'in the bottle: black, the main color of the bag: green, the text on the item: black, etc.\n'
        '3. Texts: Please extract all texts on the item with the position and color of the text. '
        'For example, "ingredients: on the surface, black". If there is no recognized text. '
        'Please only output "None".\n'
        '4. Function: Please describe the usage of the item in the given picture.\n'
        '5. Summary of the item: Please summarize the above descriptions in sentences one-by-one.\n'
    )

    model = 'qwen2.5-vl-3b'
    messages = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': 'images/002_coca-cola_soda_diet_pop_bottle.jpeg',
                },
                {
                    'type': 'text',
                    'text': prompt,
                },
            ],
        },
    ]

    model_path = models[model]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
#    model = 'internvl2_5-1b'
#    prompt = '\n\n'.join([system_prompt, prompt])
#    prompt = f"<image>\n{prompt}:"

    image = Image.open('images/002_coca-cola_soda_diet_pop_bottle.jpeg')
    message = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }

    response = call(message, schema, model=model, max_new_tokens=1000).strip()
    response = call(message, schema, model=model, max_new_tokens=1000)

