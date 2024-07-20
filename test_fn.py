import torch
torch.manual_seed(1)
from PIL import Image
import requests
import json
import os
import numpy as np
from io import BytesIO
import argparse
from tqdm import tqdm
from util.metrics import compute_iou, compute_PNLS
from util.model_handler import *


question_type_zoo = {'font_color': 'mcq',
                 'font_size': 'mcq',
                 'localize_o': 'mcq',
                 'localize_t': 'mcq',
                 'spatial_oo': 'mcq',
                 'spatial_ot': 'mcq',
                 'spatial_tt': 'mcq', 
                 'recognition_label': 'recognition_t',
                 'recognition_pos': 'recognition_t',
                 'grounding_t': 'grounding_t',
                 'grounding_o': 'grounding_o'}

model_handler_zoo = {'GPT-4o': GPT4Handler, 
                     'GPT-4V': GPT4Handler, 
                     'Phi-3-Vision': Phi3VHandler,
                     'Qwen-vl-plus': QwenHandler,
                     'Qwen-vl-max': QwenHandler,
                     'LLaVA-1.5': LLaVAHandler,
                     'LLaVA-NEXT': LLaVANextHandler,
                     'LLaVA-NEXT-34B': LLaVANext34BHandler,
                     'Monkey': MonkeyHandler,
                     'idefics-80b': IdeficsHandler,
                     'idefics-2-8b': Idefics2Handler,
                     'chameleon-7b': Chameleon7BHandler,
                     'chameleon-30b': Chameleon7BHandler,
                    }

def init_handler(model_name):
    handler_type = model_handler_zoo[model_name]
    handler = handler_type(model_name=model_name)
    return handler


    
def test_mcq(task, model_name, handler, result_dir=None):
    
    qa_dir = os.path.join('./QA/mcq', f'{task}.json')
    question_lists = json.load(open(qa_dir, 'r'))
    question_lists = question_lists[:50]
    results = []
    n_correct = 0
    pbar = tqdm(enumerate(question_lists), desc=f'task: {task}', total=len(question_lists), ncols=120)
    for i, q in pbar:
        img_id = q['img_id']
        true_answer = q['Answer']
        img_dir = f'./img/{img_id}.jpg'
        question = q['Question']

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # model specific code block
        # ----------------------------------------------------------------------------------------
        # get output from model
        model_answer = handler.ask(input_string=question, img_dir=img_dir, question_type='mcq')

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        q['model_answer'] = model_answer
        results.append(q)            
        
        n_correct += (true_answer == model_answer)
        pbar.set_postfix({'correct': n_correct, 'Answer': true_answer, 'Model': model_answer})
    
    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)

    return results


def test_recognition(task, model_name, handler, result_dir=None):
    
    qa_dir = os.path.join('./QA/recognition', f'{task}.json')
    question_lists = json.load(open(qa_dir, 'r'))
    question_lists = question_lists[:50]
    results = []
    pnls_list = []
    pbar = tqdm(enumerate(question_lists), desc=task, total=len(question_lists), ncols=120)
    for i, q in pbar:
        img_id = q['id']
        true_answer = q['answers']
        img_dir = f'./img/{img_id}.jpg'
        question = q['question']

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # model specific code block
        # ----------------------------------------------------------------------------------------
        # get output from model
        model_answer = handler.ask(input_string=question, img_dir=img_dir, question_type='recognition_t')
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        q['model_answer'] = model_answer

        # compute PNLS
        pnls = compute_PNLS(q['answers'].lower().replace('\"', ''), model_answer.lower())
        results.append(q)
        pnls_list.append(pnls)
        pbar.set_postfix({'PNLS': pnls, 'mean PNLS': np.mean(pnls_list)})
        
    s = np.mean(pnls_list)
    # print(f'mean PNLS: {s}')
    print(pnls_list)
    print(sum([1 for x in pnls_list if x > 0.9]))

    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)
    return results


def test_grounding_o(task, model_name, handler, result_dir=None):

    question_lists = json.load(open('./QA/grounding/grounding_o.json', 'r'))
    question_lists = question_lists[:50]
    
    results = []
    iou_list = []
    pbar = tqdm(enumerate(question_lists), desc='grounding_o', total=len(question_lists), ncols=120)
    for i, q in pbar:
        img_id = q['img_id']
        box, box_n = q['box']
        img_dir = f'./img/{img_id}.jpg'
        target_object = q['object']

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # model specific code block
        # ----------------------------------------------------------------------------------------
        # get output from model
        model_answer = handler.ask(input_string=target_object, img_dir=img_dir, question_type='grounding_o')
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if 'ERROR' in model_answer:
            continue
            
        q['model_answer'] = model_answer
        results.append(q)
        iou_s = compute_iou(box_n, model_answer)
        iou_list.append(iou_s)
        
        pbar.set_postfix({'valid': len(iou_list), 'mean IoU': np.mean(iou_list)})
        
    # print(f'mean IoU score: {np.mean(s)}, len: {len(s)}')

    print(iou_list)
    print(sum([1 for x in iou_list if x > 0.3]))
    
    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)
    return results


def test_grounding_t(task, model_name, handler, result_dir=None):

    question_lists = json.load(open('./QA/grounding/grounding_t.json', 'r'))
    question_lists = question_lists[:50]
    
    results = []
    iou_list = []
    pnls_list = []
    pbar = tqdm(enumerate(question_lists), desc='grounding_t', total=len(question_lists), ncols=120)
    for i, q in pbar:
        img_id = q['img_id']
        box, box_n = q['box']
        img_dir = f'./img/{img_id}.jpg'
        target_object = q['target']
        pos = q['pos']
        question = q['question']
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # model specific code block
        # ----------------------------------------------------------------------------------------
        # get output from model
        model_answer = handler.ask(input_string=question, img_dir=img_dir, question_type='grounding_t')
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if 'ERROR' in model_answer:
            continue
        else:
            text_a, box_a = json.loads(model_answer)

        q['model_answer'] = [text_a, box_a]
        results.append(q)
        iou_s = compute_iou(box_n, box_a)
        pnls = compute_PNLS(q['text'].lower().replace('\"', ''), text_a.lower())

        iou_list.append(iou_s)
        pnls_list.append(pnls)
        
        pbar.set_postfix({'valid': len(iou_list), 'mean IoU': np.mean(iou_list), 'mean PNLS': np.mean(pnls_list)})

    print(iou_list)
    print(pnls_list)
    print(sum([1 for x in iou_list if x > 0.3]))
    print(sum([1 for x in pnls_list if x > 0.9]))
        

    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)
    return results



def test(task, model_name, handler, result_dir=None):
    
    q_type = question_type_zoo[task]

    if q_type == 'recognition_t':
        test_recognition(task, model_name, handler, result_dir=result_dir)

    elif q_type == 'mcq':
        test_mcq(task, model_name, handler, result_dir=result_dir)

    elif q_type == 'grounding_o':
        test_grounding_o(task, model_name, handler, result_dir=result_dir)

    elif q_type == 'grounding_t':
        test_grounding_t(task, model_name, handler, result_dir=result_dir)





