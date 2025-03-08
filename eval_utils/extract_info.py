import base64
import json
import os

import openai_api
from io import BytesIO
from pydantic import BaseModel
from PIL import Image

class Color(BaseModel):
    description: str
    color: str

class Text(BaseModel):
    text: str
    position: str

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
#'The output format should be as follows:\n'
#'Shape: <description of shape>.\n'
#'Colors:\n'
#'<description 1>: <color 1>\n'
#'<description ...>: <color ...>\n'
#'Texts:\n'
#'<text 1>: <text position 1>\n'
#'<text ...>: <text position ...>\n'
#'Summary: <Summary>\n'
#'Name: <name of the item>\n'
#'Please strictly follow the above format. There may be multiple lines in "Colors" or "Texts". '
#'Please describe all of them line-by-line.'
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

fp = open('descriptions.jsonl', 'a+')
folder = 'images'
for f_idx, filename in enumerate(os.listdir(folder)):
    if f_idx + 1 >= 45:
        break
    basename, ext = os.path.splitext(filename)
    filename = os.path.join(folder, filename)
    base64_image = encode_image(filename)
    print(filename)

    image = Image.open(filename)
    width, height = image.size
    file_size_bytes = os.path.getsize(filename)
    print(f"Image size: {width}x{height} pixels")
    print(f"File size: {file_size_bytes/1024:.2f} KB")

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
        model='gpt-4o-mini',
        max_tokens=1000,
    )

    response_json = response.model_dump(mode='json')
    response_json['name'] = basename
    fp.write(json.dumps(response_json)+'\n')

fp.close()

