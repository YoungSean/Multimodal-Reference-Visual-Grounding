import os
import time

import anthropic
import yaml

if os.path.exists('eval_utils/keys/anthropic_key'):
    key_path = 'eval_utils/keys/anthropic_key'
elif os.path.exists('keys/anthropic_key'):
    key_path = 'keys/anthropic_key'
with open(key_path, 'r') as f:
    api_key = f.read().strip()

client = anthropic.Anthropic(api_key=api_key)

def call(messages, tools, model="claude-3-haiku-20240307", max_tokens=500):
    system_msg = ''
    if 'system' == messages[0]['role']:
        system_msg = messages[0]['content'][0]['text']
        messages = messages[1:]
    while True:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,
                system=system_msg,
                messages=messages,
                tools=tools,
                tool_choice={"type": "tool", "name": tools[0]['name']},
            )
            break
        except Exception as e:
            print('Errrrrrrrrrrrrrrrrrr', str(e))
            time.sleep(60)
    
    prediction = response.content[0].input

    return prediction
    
if __name__ == '__main__':
    pass

