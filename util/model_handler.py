import json
from PIL import Image 
from http import HTTPStatus
import dashscope
import torch
import random
import re

def extract_choice(input_string):
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', input_string)
    # Convert the list of numbers from strings to integers
    numbers = [int(number) for number in numbers]

    if len(numbers) > 0:
        return numbers[0]
    elif len(numbers)  == 0:
        return 0

torch.manual_seed(1)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Base handler

class BaseModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self, model_name: str):
        raise Exception("Not Implemented")
    
    def ask(self, input_string: str, img_dir: str, question_type: str):
        raise Exception("Not Implemented")

    def prompt_wrapper(self, input_string, question_type):
        templates = {
            'grounding_o': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text only.'),
            'grounding_t': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text and bounding box only. For example, "Hello world", [x_min, y_min, x_max, y_max]'),
        
            'mcq': (f'{input_string}\nOnly print the index of the correct choice as answer, such as 1, 2, 3, or 4'),
            'recognition_t': (f'{input_string}\nOnly print the text; do not include any other descriptions.')
        }

        return templates.get(question_type, input_string)
        
    def load_img(self, img_input):
        if img_input.startswith('http://') or img_input.startswith('https://'):
            # Handle URL
            response = requests.get(img_input)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Handle local file path
            image = Image.open(img_input).convert('RGB')

        return image
        
    def auto_extract(self, question_type, reply):
        try:
            if question_type == 'grounding_o':
                reply = reply.replace(", ", ",").replace(",", ", ")
                reply = json.loads(reply)
                assert len(reply) == 4
                
            elif question_type == 'grounding_t':
                text_a = reply.split('[')[0].replace('\"', '').strip()
                box_a = '[' + reply.split('[')[1]
                box_a = json.loads(box_a)
                box_a = [float(x) for x in box_a]
                assert len(box_a) == 4
                reply = json.dumps([text_a, box_a])
                    
            elif question_type == 'mcq':
                # reply = int(reply.replace('\"', '').strip())
                numbers = re.findall(r'\d+', reply)

                if len(numbers) > 0:
                    return numbers[0]
                elif len(numbers)  == 0:
                    return 0

            else:
                reply = reply.replace('\"', '')
        except:
            reply = 'ERROR: invalid format'
            
        return reply
    
    def __repr__(self):
        return f"<Model Handler for: {self.model_name}>"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for GPT-4V and GPT-4o

class GPT4Handler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name in {'GPT-4o', 'GPT-4V'}

        self.load_model(model_name)

        self.system_prompts = {
            "grounding_o": (
                'You are an expert AI model in detecting objects in images. '
                'Please write the bounding box of the object as a python list, '
                'the coordinates are in float numbers on a scale from 0 to 1, '
                'the list consists of the four coordinates: [x_min, y_min, x_max, y_max]. '
                'Please just write the list without any explanations. A demo reply is:\n'
                '[607.99, 629.82, 924.85, 762.85]'
            ),
            "grounding_t": (
                'You are an expert in detecting objects in images. '
                'Please extract the text and its bounding box as a python list, '
                'the coordinates are in float numbers on a scale from 0 to 1, '
                'the list consists of the four coordinates: [x_min, y_min, x_max, y_max]. '
                'Please just write the list without any explanations. A demo reply is:\n'
                '"Hello world", [0.19, 0.32, 0.85, 0.76]'
            ),
            "mcq": (
                'You are an expert AI model in analyzing images. Please choose the correct answer.'
            ),
            "recognition_t": (
                'You are an expert AI model in reading text in images. Please extract the required text.'
            )
        }

    def load_model(self, model_name: str):
        if model_name == 'GPT-4o':
            from util.gpt4o import request_gpt4o
            self.ask_gpt = request_gpt4o
        elif model_name == 'GPT-4V':
            from util.gpt4v import request_gpt4v
            self.ask_gpt = request_gpt4v


    def ask(self, input_string: str, img_dir: str, question_type: str):
        sys_prompt = self.system_prompts[question_type]
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)

        model_answer = self.ask_gpt(prompt=prompt, image_path_list=[img_dir], 
                              system_prompt=sys_prompt)
        
        model_answer = self.auto_extract(question_type, model_answer)
        
        return model_answer
    

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for Claude 3.5 Sonnet

class ClaudeHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'Claude-3.5-Sonnet'

        self.load_model(model_name)

        self.system_prompts = {
            "grounding_o": (
                'You are an expert AI model in detecting objects in images. '
                'Please write the bounding box of the object as a python list, '
                'the coordinates are in float numbers on a scale from 0 to 1, '
                'the list consists of the four coordinates: [x_min, y_min, x_max, y_max]. '
                'Please just write the list without any explanations. A demo reply is:\n'
                '[607.99, 629.82, 924.85, 762.85]'
            ),
            "grounding_t": (
                'You are an expert in detecting objects in images. '
                'Please extract the text and its bounding box as a python list, '
                'the coordinates are in float numbers on a scale from 0 to 1, '
                'the list consists of the four coordinates: [x_min, y_min, x_max, y_max]. '
                'Please just write the list without any explanations. A demo reply is:\n'
                '"Hello world", [0.19, 0.32, 0.85, 0.76]'
            ),
            "mcq": (
                'You are an expert AI model in analyzing images. Please choose the correct answer.'
            ),
            "recognition_t": (
                'You are an expert AI model in reading text in images. Please extract the required text.'
            )
        }

    def load_model(self, model_name: str):
        from util.claude import request_claude
        self.ask_gpt = request_claude


    def ask(self, input_string: str, img_dir: str, question_type: str):
        sys_prompt = self.system_prompts[question_type]
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)

        model_answer = self.ask_gpt(prompt=prompt, image_path_list=[img_dir], 
                              system_prompt=sys_prompt)
        
        model_answer = self.auto_extract(question_type, model_answer)
        
        return model_answer


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for Phi-3-Vision

class Phi3VHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'Phi-3-Vision'
        self.load_model()


    def load_model(self):
        from transformers import AutoModelForCausalLM 
        from transformers import AutoProcessor 
        model_id = "microsoft/Phi-3-vision-128k-instruct" 
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


    def ask(self, input_string: str, img_dir: str, question_type: str):
        
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)

        prompt = (f'A chat between a curious human and an artificial intelligence assistant. '
              f'The assistant gives helpful, detailed, and polite answers to the human\'s questions. '
              f'USER: {prompt} ASSISTANT:')

        # setup message
        # we do zero-shot test, so ignore demonstrations
        messages = [
        # {"role": "user", "content": f"<|image_1|>\n{demo_q}"}, 
        # {"role": "assistant", "content": f"{demo_a}"}, 
        {"role": "user", "content": f"<|image_1|>\n{prompt}"} 
        ]

        # load image from dir
        image = self.load_img(img_dir)
        
        prompt_in = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt_in, [image], return_tensors="pt").to("cuda")
        
        generation_args = { 
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        model_answer = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for Qwen-vl-max and Qwen-vl-plus

class QwenHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name in {'Qwen-vl-plus', 'Qwen-vl-max'}
        self.load_model()

    def load_model(self):
        from util.API_KEYS import Qwen_API_KEY
        dashscope.api_key = Qwen_API_KEY
        self.checkpoint = self.model_name.lower()

    def prompt_wrapper(self, input_string, question_type):

        if question_type == 'grounding_o':
            question = (f'{input_string}\nProvide me with the x_min, y_min, x_max and y_max coordinate '
                        f'in range of 0 to 1 as a python list. Please just output the list.')
    
    
        elif question_type == 'grounding_t' and self.model_name == 'Qwen-vl-plus':
    
            # for qwen-vl-plus
            question = (f'{input_string} Provide me with the x_min, y_min, x_max and y_max coordinate '
                        f'in range of 0 to 1 as a python list. '
                        f'Please output the text first and then the list in the format of: text, [x1, y1, x2, y2]')

        elif question_type == 'grounding_t' and self.model_name == 'Qwen-vl-max':
            # for qwen-vl-max
            question = (f'{input_string} Provide me with the x_min, y_min, x_max and y_max coordinate '
                        f'in range of 0 to 1 as a python list. '
                        f'Please output both the text and the list. An example is: "text text", [x1, y1, x2, y2]')
    
            
        elif question_type == 'single_choice':
            question = f'{input_string}\nOnly print the index of the correct choice as answer, such as 1, 2, 3, or 4'
        elif question_type == 'recognition':
            question = f'{input_string}\nOnly print the text; do not include any other descriptions.'
        else:
            question = input_string
    
        prompt = question
        
        return prompt

    def ask(self, input_string: str, img_dir: str, question_type: str, show_usage=False):
        
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        prompt = (f'A chat between a curious human and an artificial intelligence assistant. '
              f'The assistant gives helpful, detailed, and polite answers to the human\'s questions. '
              f'USER: {prompt} ASSISTANT:')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": img_dir},
                    {"text": prompt}
                ]
            }
        ]
        response = dashscope.MultiModalConversation.call(model=self.checkpoint,
                                                         messages=messages)
        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response.status_code == HTTPStatus.OK:
            reply = response.output['choices'][0]['message']['content'][0]['text']
            if show_usage:
                usage = response.usage
                print(usage)

            reply = self.auto_extract(question_type, reply)
            return reply
        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.
            return 'unknown error'


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for PaliGemma

class PaliGemmaHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'PaliGemma'

        self.load_model()

    def load_model(self):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        from PIL import Image
        from huggingface_hub import login
        token = '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'
        login(token)
        
        model_id = "google/paligemma-3b-mix-448"
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="cuda")
        self.processor = AutoProcessor.from_pretrained(model_id)


    def ask(self, input_string: str, img_dir: str, question_type: str):
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        
        raw_image = self.load_img(img_dir)
        inputs = self.processor(prompt, raw_image, return_tensors="pt").to('cuda')
        output = self.model.generate(**inputs, max_new_tokens=4096)
    
        model_answer = self.processor.decode(output[0], skip_special_tokens=True)[len(prompt):]

        model_answer = model_answer.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)
        
        return model_answer
        



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for InternVL2-8B

class InternVLHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'InternVL2-8B'

        self.load_model()

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        path = f'OpenGVLab/InternVL2-8B'
        # path = f'OpenGVLab/InternVL2-Llama3-76B'
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    def ask(self, input_string: str, img_dir: str, question_type: str):
        from util.InternVL_util import load_image
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)

        # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        image = load_image(img_dir, max_num=20).to(torch.bfloat16).cuda()
        pixel_values = torch.cat([image], dim=0)

        question = f'<image>\n{prompt}'
        model_answer, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                       history=None, return_history=True)
        model_answer = model_answer.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)
        
        return model_answer
        

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for InternVL2-76B

class InternVL76BHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'InternVL2-76B'

        self.load_model()

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        
        def split_model(model_name):
            device_map = {}
            world_size = torch.cuda.device_count()
            num_layers = {
                'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
                'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
            # Since the first GPU will be used for ViT, treat it as half a GPU.
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
            device_map['vision_model'] = 0
            device_map['mlp1'] = 0
            device_map['language_model.model.tok_embeddings'] = 0
            device_map['language_model.model.embed_tokens'] = 0
            device_map['language_model.output'] = 0
            device_map['language_model.model.norm'] = 0
            device_map['language_model.lm_head'] = 0
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

            return device_map
            
        # path = 'OpenGVLab/InternVL2-Llama3-76B'
        # device_map = split_model('InternVL2-Llama3-76B')

        path = 'OpenGVLab/InternVL2-40B'
        device_map = split_model('InternVL2-40B')
        
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


    def ask(self, input_string: str, img_dir: str, question_type: str):
        from util.InternVL_util import load_image
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)

        # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        pixel_values = load_image(img_dir, max_num=12).to(torch.bfloat16).cuda()

        question = f'<image>\n{prompt}'
        model_answer, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                       history=None, return_history=True)
        model_answer = model_answer.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)
        
        return model_answer


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for Chameleon-7B

class Chameleon7BHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'chameleon-7b'
        self.load_model()

    def load_model(self):
        from chameleon.inference.chameleon import ChameleonInferenceModel
        self.model = ChameleonInferenceModel(
            "/sensei-fs/users/jianc/data/models/7b/",
            "/sensei-fs/users/jianc/data/tokenizer/text_tokenizer.json",
            "/sensei-fs/users/jianc/data/tokenizer/vqgan.yaml",
            "/sensei-fs/users/jianc/data/tokenizer/vqgan.ckpt",
        )

    def ask(self, input_string: str, img_dir: str, question_type: str):

        
        # prepare prompt
        prompt_text = self.prompt_wrapper(input_string=input_string, question_type=question_type)

        tokens = self.model.generate(
        prompt_ui=[
            {"type": "image", "value": f"file:{img_dir}"},
            {"type": "text", "value": f"{prompt_text}"},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
            ]
        )

        reply = self.model.decode_text(tokens)[0]
        print('>' * 100)
        print(reply)
        print('>' * 100)
        model_answer = reply.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for LLaVA-1.5

class LLaVAHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'LLaVA-1.5'
        self.load_model()

    def load_model(self):
        from transformers import pipeline
        model_id = "llava-hf/llava-1.5-13b-hf"
        self.pipe = pipeline("image-to-text", model=model_id, device_map="auto")

    def ask(self, input_string: str, img_dir: str, question_type: str):
        
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        prompt = (f'"USER: <image>\n<{prompt}> ASSISTANT:')
        # load image from dir
        image = self.load_img(img_dir)

        reply = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        model_answer = reply[0]['generated_text'].split(' ASSISTANT:')[-1]
        model_answer = model_answer.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for LLaVA-NEXT (LLaVA-1.6)

class LLaVANextHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'LLaVA-NEXT'
        self.load_model()

    def load_model(self):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf",
                                                                  torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    def prompt_wrapper(self, input_string, question_type):
        templates = {
            'grounding_o': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text only.'),
            'grounding_t': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text first and then the bounding box. For example, "Hello world" [x_min, y_min, x_max, y_max]'),
        
            'mcq': (f'{input_string}\nOnly print the index of the correct choice as answer, such as 1, 2, 3, or 4'),
            'recognition_t': (f'{input_string}\nOnly print the text; do not include any other descriptions.')
        }

        return templates.get(question_type, input_string)

    def ask(self, input_string: str, img_dir: str, question_type: str):
        
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        prompt = (f'A chat between a curious human and an artificial intelligence assistant. '
              f'The assistant gives helpful, detailed, and polite answers to the human\'s questions. '
              f'USER: <image>\n {prompt} ASSISTANT:')
        # load image from dir
        image = self.load_img(img_dir)

        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda")
        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=100)
    
        reply = self.processor.decode(output[0], skip_special_tokens=True)
        model_answer = reply.split('ASSISTANT:')[-1]
        model_answer = model_answer.replace('\"', '')
        
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for LLaVA-NEXT 34b

class LLaVANext34BHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'LLaVA-NEXT-34B'
        self.load_model()

    def load_model(self):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf",
                                                                  torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    def prompt_wrapper(self, input_string, question_type):
        templates = {
            'grounding_o': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text only.'),
            'grounding_t': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text first and then the bounding box. For example, "Hello world", [x_min, y_min, x_max, y_max]'),
        
            'mcq': (f'{input_string}\nOnly print the index of the correct choice as answer, such as 1, 2, 3, or 4'),
            'recognition_t': (f'{input_string}\nOnly print the text; do not include any other descriptions.')
        }

        return templates.get(question_type, input_string)

    def ask(self, input_string: str, img_dir: str, question_type: str):
        
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        prompt = (f'<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n'
                  f'{prompt}<|im_end|><|im_start|>assistant\n')

        # prompt = (f'A chat between a curious human and an artificial intelligence assistant. '
        #       f'The assistant gives helpful, detailed, and polite answers to the human\'s questions. '
        #       f'USER: <image>\n {prompt} ASSISTANT:')
        # load image from dir
        image = self.load_img(img_dir)

        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda")
        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=100)
    
        reply = self.processor.decode(output[0], skip_special_tokens=True)
        model_answer = reply.split('assistant\n')[-1]
        model_answer = model_answer.replace('\"', '')
        
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for Monkey-Chat

class MonkeyHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'Monkey'
        self.load_model()

    def load_model(self):
        
        from modelscope import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        checkpoint = "echo840/Monkey-Chat"
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id


    def ask(self, input_string: str, img_dir: str, question_type: str):
        
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        prompt = (f'A chat between a curious human and an artificial intelligence assistant. '
              f'The assistant gives helpful, detailed, and polite answers to the human\'s questions. '
              f'USER: {prompt} ASSISTANT:')
        
        prompt = f'<img>{img_dir}</img> {prompt} Answer: '
        # load image from dir


        input_ids = self.tokenizer(prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        pred = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                    )
        model_answer = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True)
        model_answer.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer
        
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for IDEFICS-1 80B

class IdeficsHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'idefics-80b'
        self.load_model()

    def load_model(self):
        # Use a pipeline as a high-level helper
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def prompt_wrapper(self, input_string, question_type):
        templates = {
            'grounding_o': (f'{input_string}\n'
                            f'Please write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text only.'),
            'grounding_t': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text and bounding box only. For example, "Hello world", [x_min, y_min, x_max, y_max]'),
        
            'mcq': (f'{input_string}\nOnly print the index of the correct choice as answer, such as 1, 2, 3, or 4. Do not output additional description.'),
            'recognition_t': (f'{input_string}\nOnly print the text; do not include any other descriptions.')
        }

        return templates.get(question_type, input_string)

    def ask(self, input_string: str, img_dir: str, question_type: str):
        
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)

        prompts = [[f"User:", f"{img_dir}", f"{prompt}<end_of_utterance>", "\nAssistant:"]]

        # --batched mode
        inputs = self.processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        # inputs = self.processor(prompt, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        
        # Generation args
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        
        generated_ids = self.model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=500)
        generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=500)

        reply = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
            
        model_answer = reply[0].split('Assistant:')[-1]
        # print(model_answer)
        model_answer = model_answer.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for IDEFICS-2 8B

class Idefics2Handler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        assert model_name == 'idefics-2-8b'
        self.load_model()

    def load_model(self):
        # Use a pipeline as a high-level helper
        from transformers import AutoProcessor, AutoModelForVision2Seq
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        self.model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b", device_map="auto")


    def prompt_wrapper(self, input_string, question_type):
        templates = {
            'grounding_o': (f'{input_string}\n'
                            f'Please write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text only.'),
            'grounding_t': (f'{input_string}\nPlease write the position as a bounding box, '
                            f'and output the [x_min, y_min, x_max, y_max] coordinates in float numbers in python list.\n'
                            f'Output the text and bounding box only. For example, "Hello world", [x_min, y_min, x_max, y_max]'),
        
            'mcq': (f'{input_string}\nOnly print the index of the correct choice as answer, such as 1, 2, 3, or 4. Do not output additional description.'),
            'recognition_t': (f'{input_string}\nOnly print the text; do not include any other descriptions.')
        }

        return templates.get(question_type, input_string)
        
    def ask(self, input_string: str, img_dir: str, question_type: str):

        from transformers.image_utils import load_image
        # prepare prompt
        prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        image = load_image(img_dir)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompt}"},
                    ]
                },     
            ]
        prompt_1 = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_1, images=[image], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        reply = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        model_answer = reply[0].split('Assistant:')[-1]
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer

if __name__=="__main__":

    M = BaseModelHandler('GPT_4')
    p = M.prompt_wrapper('hahahhha', question_type='adada')
    print(p)
