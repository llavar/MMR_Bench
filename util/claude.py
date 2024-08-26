import os
from openai import AzureOpenAI
from openai import OpenAI
import base64
import requests
from io import BytesIO
from PIL import Image

# Set the environment variable
os.environ['OPENAI_API_KEY'] = "sk-BHVOYqkc__nwbwxN1eNApQ"
os.environ['OPENAI_BASE_URL'] = "https://llm-proxy.dcai.corp.adobe.com/"

client = OpenAI()

lm_id = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"



def encode_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    # image = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
    # r = 512 / max(image.size[0], image.size[0])
    # image = image.resize((int(r * image.size[0]), int(r * image.size[1])))

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8'), image


def prepare_img_input(image_path_list):
    image_input = []
    for image_path in image_path_list:
        base64_image, _ = encode_image(image_path)
        image_input.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
        })
    return image_input


def prepare_prompt(prompt, image_path_list, system_prompt='You are an AI visual assistant', demonstration=None):

    # append system prompt at the first entry
    input_msgs = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},]

    # Optional, append demo visual-question-answer pairs
    if demonstration is not None:
        for demo_q, demo_url_list, demo_a in demonstration:
            input_msgs.append({
                "role": "user",
                "content": [{"type": "text", "text": demo_q}] + prepare_img_input(demo_url_list)
            })
            input_msgs.append({"role": "assistant", "content": [{"type": "text", "text": demo_a}]})

    # append question and image url to ask append multiple images
    input_msgs.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + prepare_img_input(image_path_list)
    })

    return input_msgs


def request_claude(prompt, image_path_list, system_prompt):

    input_msgs = prepare_prompt(prompt, image_path_list, system_prompt)

    response = client.chat.completions.create(
            model=lm_id,
            # messages=[
            #     # {"role": "user", "content": ""},
            #     {"role": "user",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": prompt
            #         },
            #     ],
            #     },
            # ],
            messages=input_msgs,
            max_tokens=4096,
            temperature=0.0,
        )
    output = response.choices[0].message.content
    
    return output




if __name__ == "__main__":

    prompt = 'Please describe the image.'
    image_path = ['https://puar-playground.github.io/assets/img/covers/HP.png']
    cap = request_claude(prompt, image_path, system_prompt='You are an AI visual assistant')

    print('image_path:', image_path)
    print('gpt answer:', cap)


