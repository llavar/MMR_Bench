import base64
import requests
from io import BytesIO
from PIL import Image
from util.API_KEYS import AzureOpenAI_gpt4v_KEY, AzureOpenAI_gpt4v_API_BASE


headers = {
    "Content-Type": "application/json",
    "api-key": AzureOpenAI_gpt4v_KEY
}


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



def prepare_prompt_4V(prompt, image_path_list, system_prompt, context_list=None):

    image_input = []
    for image_path in image_path_list:
        base64_image, _ = encode_image(image_path)
        image_input.append({
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                # "url": image_path
                }
            })

    input_msgs = [{"role": "system", "content": [system_prompt]}]

    if context_list is not None:
        for context in context_list:
            context_q, context_url, context_a = context
            base64_image, _ = encode_image(context_url)
            input_msgs.append({"role": "user", "content": [{"type": "text", "text": context_q},
                                                           {"type": "image_url",
                                                            "image_url":
                                                                {"url": f"data:image/jpeg;base64,{base64_image}"}}]})
            input_msgs.append({"role": "assistant", "content": [{"type": "text", "text": context_a}]})

    input_msgs.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + image_input
        })

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": input_msgs,
        "max_tokens": 800
    }

    return payload


def request_gpt4v(prompt, image_path_list, system_prompt, context_list=None):

    deployment_name = 'gpt-4-vision-preview'
    base_url = f"{AzureOpenAI_gpt4v_API_BASE}openai/deployments/{deployment_name}"
    # Prepare endpoint, headers, and request body
    endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview"

    try:
        payload = prepare_prompt_4V(prompt=prompt, image_path_list=image_path_list,
                                    system_prompt=system_prompt, context_list=context_list)
    except:
        return 'ERROR: image encoding'
        
    while True:
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            res = response.json()['choices'][0]['message']['content']
            return res
        except:
            continue
            
    return 'ERROR: GPT-4V no response'


if __name__ == "__main__":

    prompt = 'Please describe the image.'
    image_path = ['https://puar-playground.github.io/assets/img/covers/HP.png']
    cap = request_gpt4v(prompt, image_path)

    print('image_path:', image_path)
    print('gpt answer:', cap)



