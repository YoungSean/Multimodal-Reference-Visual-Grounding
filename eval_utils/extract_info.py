import base64
import json
import os

from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--extraction-model', help='extraction model: gpt-4o, gpt-4o-mini, claude-3-haiku, internvl2_5-xb, qwen2.5-vl-xb')
args = parser.parse_args()

if 'gpt-4' == args.extraction_model[:5]:
    import openai_api
elif 'claude' == args.extraction_model[:6]:
    import anthropic_api
elif 'internvl' == args.extraction_model[:8] or 'qwen' == args.extraction_model[:4]:
    import vllm_api

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
#    name: str

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
#'6. Name the item: assign a name based on the above information.\n'
#'You are given an image of an item on a flat surface (on a table, ground, etc.). '
#'Please first carefully read and understand the image in detail. '
#'If there are multiple items, only carefully look through one of them. '
#'Then, describe the item in detail by following the steps and format below.\n'
#'1. Shape: Please describe the shape or type of the item, such as a bottle, '
#'bag, round item, square item, etc.\n'
#'2. Colors: Please describe all the colors on or in the item, such as label colors, '
#'cap colors, cover colors, etc. The item may be covered by multiple colors. '
#'Please describe all of them one by one.\n'
#'3. Texts: Please extract all texts on the item with the position and color of the text. '
#'Position is the relative position on the item but not BBOX numbers on the image. '
#'If there is no recognized text. Please only output "None".\n'
#'4. Function: Please describe the usage of the item in the given picture.\n'
#'5. Summary of the item: Please summarize the above descriptions in a few sentences.\n'
#'6. Name the item: assign a name based on the above information.\n'
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

fp = open(f'results/descriptions_{args.extraction_model}.jsonl', 'w+')

def openai_extraction(base64_image, model):
    messages = [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': system_prompt,
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt,
                },
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/{ext};base64, {base64_image}'},
                }
            ],
        },
    ]

    response = openai_api.call_json(
        messages,
        response_format=Extraction,
        model=model,
        max_tokens=3000,
    )
    response_json = response.model_dump(mode='json')
    response_json['filename'] = basename
    return response_json

def anthropic_extraction(base64_image, ext, model):
    tools = [
        {
            "name": "record_summary",
            "description": "Record summary of an image using well-structured JSON.",
            "input_schema": Extraction.model_json_schema(mode='validation'),
        },
    ]
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f'image/{ext[1:]}',
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]
    response = anthropic_api.call(
        messages,
        tools,
        model=model,
        max_tokens=1000,
    )
    response['filename'] = basename
    return response

def qwenvl_extraction(image, image_path, model):
    model_path = vllm_api.get_model_path(model)

    if 'qwen-vl' == model[:7]:
        messages = [
            {'image': image_path},
            {'text': '\n\n'.join([system_prompt, prompt])},
        ]

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        formatted_prompt = tokenizer.from_list_format(messages)
    else:
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
                        'image': image_path,
                    },
                    {
                        'type': 'text',
                        'text': prompt,
                    },
                ],
            },
        ]

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    message = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": image},
    }
    schema = Extraction.model_json_schema(mode='validation')

    response_json = vllm_api.call(
        message,
        schema,
        model=model,
        max_new_tokens=2000,
    )
    if not isinstance(response_json, str):
        response_json['filename'] = basename
    else:
        response_json = {
            'content': response_json,
            'filename': basename,
        }
    return response_json

def internvl_extraction(image, model):
    whole_prompt = '\n\n'.join([system_prompt, prompt])
    formatted_prompt = f"<image>\n{whole_prompt}"
    message = {
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": image},
    }
    schema = Extraction.model_json_schema(mode='validation')
    response_json = vllm_api.call(
        message,
        schema,
        model=model,
        max_new_tokens=2000,
    )
    if not isinstance(response_json, str):
        response_json['filename'] = basename
    else:
        response_json = {
            'content': response_json,
            'filename': basename,
        }
    return response_json

unformatted_count = 0
folder = 'template_images'
for f_idx, filename in tqdm(enumerate(os.listdir(folder)), total=len(os.listdir(folder))):
#    if f_idx + 1 <= 60:
#        continue
#    if f_idx + 1 >= 5:
#        break
    basename, ext = os.path.splitext(filename)
    filename = os.path.join(folder, filename)
    base64_image = encode_image(filename)
    print(filename)

    file_size_bytes = os.path.getsize(filename)
    image = Image.open(filename)
    if 'jpeg' != ext or 'jpg' != ext:
        image = image.convert('RGB')
    width, height = image.size
    print(f"Image size: {width}x{height} pixels")
    print(f"File size: {file_size_bytes/1024:.2f} KB")

    print(args.extraction_model)
    if 'gpt-4' == args.extraction_model[:5]:
        response_json = openai_extraction(base64_image, args.extraction_model)
    elif 'claude' == args.extraction_model[:6]:
        response_json = anthropic_extraction(base64_image, ext, args.extraction_model)
    elif 'internvl' == args.extraction_model[:8]:
        response_json = internvl_extraction(image, args.extraction_model)
    elif 'qwen' == args.extraction_model[:4]:
        response_json = qwenvl_extraction(image, filename, args.extraction_model)
#    print(response_json)

    if isinstance(response_json, str):
        unformatted_count += 1

    fp.write(json.dumps(response_json)+'\n')

fp.close()
print(f'{args.extraction_model}: Unformatted Rate: {unformatted_count / len(os.listdir(folder))}')

