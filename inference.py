from util.argparse_format import MMR_args
import torch
torch.manual_seed(1)
from test_fn import test, init_handler
import os
import json
from datasets import load_dataset

CLAIM = ('Proprietary models like GPT-4o and Claude-3.5-Sonnet generate stochastic answers and may undergo internal updates, '
         'leading to results that may differ from those reported in the paper.')

task_zoo = ['font_size', 'font_color', 'localize_o', 'localize_t', 'spatial_ot', 'spatial_oo', 'spatial_tt', 
            'recognition_label', 'recognition_pos', 'grounding_o', 'grounding_t',]

model_zoo = ['GPT-4V', 'GPT-4o', 'Claude-3.5-Sonnet', 'Phi-3-Vision', 'Qwen-vl-plus', 'Qwen-vl-max', 'LLaVA-NEXT', 'LLaVA-NEXT-34B',
              'LLaVA-1.5', 'Monkey', 'idefics-80b', 'idefics-2-8b', 'chameleon-7b', 'chameleon-30b']


if __name__ == "__main__":

    os.system('clear')
    print(CLAIM)

    assert MMR_args.model in model_zoo
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

    print(f'Testing: {MMR_args.model}')
    handler = init_handler(model_name=MMR_args.model)

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
    
    





