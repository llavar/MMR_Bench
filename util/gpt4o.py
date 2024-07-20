from openai import AzureOpenAI
import base64
import requests
from io import BytesIO
from PIL import Image
from util.API_KEYS import AzureOpenAI_gpt4o_KEY, AzureOpenAI_gpt4o_ENDPOINT

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


def prepare_prompt_4o(prompt, image_path_list, system_prompt='You are an AI visual assistant', demonstration=None):

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

def request_gpt4o(prompt, image_path_list, system_prompt='You are an AI visual assistant', demonstration=None):

    client = AzureOpenAI(
        azure_endpoint=AzureOpenAI_gpt4o_ENDPOINT,
        api_version="2024-02-15-preview",
        api_key=AzureOpenAI_gpt4o_KEY
    )

    msg = prepare_prompt_4o(prompt, image_path_list, system_prompt=system_prompt, demonstration=demonstration)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=msg,
        max_tokens=800,
    )
    output = response.choices[0].message.content
    return output


if __name__ == "__main__":

    # construct demonstrations
    demo_prompt = 'Please list attribute of provided images.'
    demo_urls = ['https://1000logos.net/wp-content/uploads/2021/05/Google-logo.png',
                 'https://miro.medium.com/v2/resize:fit:4800/format:webp/0*c-PJKeN6JqEUKyZ8.png']
    demo_answer = ('1 {company: Google, content: text}\n'
                   '2 {company: OpenAI, content: logo and text}')

    demonstration = [[demo_prompt, demo_urls, demo_answer]]

    # image to ask question
    prompt = 'Please list attribute of provided images.'
    image_url_list = ['https://1000logos.net/wp-content/uploads/2021/04/Adobe-logo.png',
                      'https://blog.logomyway.com/wp-content/uploads/2021/11/meta-logo-blue.jpg']

    reply = request_gpt4o(prompt, image_url_list, demonstration=demonstration)
    print(reply)


