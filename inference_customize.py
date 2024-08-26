from util.argparse_format import MMR_args
import torch
torch.manual_seed(1)
import os
import json
from util.model_handler import BaseModelHandler
from datasets import load_dataset

task_zoo = ['font_size', 'font_color', 'localize_o', 'localize_t', 'spatial_ot', 'spatial_oo', 'spatial_tt', 
            'recognition_label', 'recognition_pos', 'grounding_o', 'grounding_t',]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for new model

class MyHandler(BaseModelHandler):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.load_model()

    def load_model(self):
        from transformers import AutoModelForCausalLM 
        from transformers import AutoProcessor 
        model_id = "usr_name/SOTA_model"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def ask(self, input_string: str, img_dir: str, question_type: str):
        
        # prepare prompt
        question_prompt = self.prompt_wrapper(input_string=input_string, question_type=question_type)
        # load image from dir
        image = self.load_img(img_dir)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # # Do your magic to generate text response
        # prompt = some_prompt + question_prompt
        # response = self.model(image, prompt)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        model_answer = model_answer.replace('\"', '').strip()
        model_answer = self.auto_extract(question_type, model_answer)

        return model_answer



if __name__ == "__main__":

    os.system('clear')

    assert MMR_args.task in task_zoo + ['all']
    assert MMR_args.device in {'auto', 'cuda', 'cpu'}

    # check result folder existence
    if MMR_args.result_dir is not None:
        if not os.path.isdir(MMR_args.result_dir):
            os.mkdir(MMR_args.result_dir)


    MMR_data = load_dataset('puar-playground/MMR', split='test')

    # setup device
    if MMR_args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = MMR_args.device

    print(f'Testing using a new model')
    handler = MyHandler(model_name='my_new_model')


    if MMR_args.task != 'all':
        qa_data = [x for x in MMR_data if x['task']==MMR_args.task]
        test(task=MMR_args.task, qa_data=qa_data, model_name=MMR_args.model, handler=handler, result_dir=MMR_args.result_dir)
        print(f'{MMR_args.model} score on MMR benchmark for {MMR_args.task} is: {score}')

    else:
        score = 0
        for task in task_zoo:
            qa_data = [x for x in MMR_data if x['task']==task]
            score_task = test(task=task, qa_data=qa_data, model_name=MMR_args.model, handler=handler, result_dir=MMR_args.result_dir)

            score += score_task
            print(f'{MMR_args.model} score on MMR benchmark for {task} is: {score_task}')
            
        
        print(f'{MMR_args.model} total score on MMR benchmark is: {score}')
    
    





