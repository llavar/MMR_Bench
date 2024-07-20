from util.argparse_format import MMR_args
import torch
torch.manual_seed(1)
from test_fn import test, init_handler
import os


task_zoo = ['font_size', 'font_color', 'localize_o', 'localize_t', 'spatial_ot', 'spatial_oo', 'spatial_tt', 
            'recognition_label', 'recognition_pos', 'grounding_o', 'grounding_t',]

# task_zoo = ['grounding_o', 'grounding_t',]


model_zoo = ['GPT-4V', 'GPT-4o', 'Phi-3-Vision', 'Qwen-vl-plus', 'Qwen-vl-max', 'LLaVA-NEXT', 'LLaVA-NEXT-34B',
              'LLaVA-1.5', 'Monkey', 'idefics-80b', 'idefics-2-8b', 'chameleon-7b', 'chameleon-30b']


if __name__ == "__main__":

    os.system('clear')

    assert MMR_args.model in model_zoo
    assert MMR_args.task in task_zoo + ['all']
    assert MMR_args.device in {'auto', 'cuda', 'cpu'}
    
    # setup device
    if MMR_args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = MMR_args.device

    print(f'Testing: {MMR_args.model}')
    handler = init_handler(model_name=MMR_args.model)

    if MMR_args.task != 'all':
        
        test(task=MMR_args.task, model_name=MMR_args.model, handler=handler, result_dir=None)

    else:
        for task in task_zoo:
            test(task=task, model_name=MMR_args.model, handler=handler, result_dir=None)
    
    











