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
                     'Claude-3.5-Sonnet': ClaudeHandler,
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
    handler_class = model_handler_zoo[model_name]
    handler = handler_class(model_name=model_name)
    return handler



def test_mcq(task, qa_data, model_name, handler, result_dir=None):
    
    results = []
    score = 0
    pbar = tqdm(enumerate(qa_data), desc=task, total=len(qa_data), ncols=120)
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
        
        score += (true_answer == model_answer)
        pbar.set_postfix({'score': score, 'Answer': true_answer, 'Model': model_answer})
    
    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)

    return score


def test_recognition(task, qa_data, model_name, handler, result_dir=None):
    
    results = []
    pnls_list = []
    pbar = tqdm(enumerate(qa_data), desc=task, total=len(qa_data), ncols=120)
    for i, q in pbar:
        img_id = q['img_id']
        question = q['Question']
        img_dir = f'./img/{img_id}.jpg'

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # model specific code block
        # ----------------------------------------------------------------------------------------
        # get output from model
        model_answer = handler.ask(input_string=question, img_dir=img_dir, question_type='recognition_t')
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        q['model_answer'] = model_answer

        # compute PNLS
        pnls = compute_PNLS(q['Answer'].lower().replace('\"', ''), model_answer.lower())
        results.append(q)
        pnls_list.append(pnls)
        score = sum([1 for x in pnls_list if x > 0.9])
        pbar.set_postfix({'score': score})


    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)
    return score


def test_grounding_o(task, qa_data, model_name, handler, result_dir=None):

    results = []
    iou_list = []
    pbar = tqdm(enumerate(qa_data), desc='grounding_o', total=len(qa_data), ncols=120)
    for i, q in pbar:
        img_id = q['img_id']
        true_box = q['Answer']
        img_dir = f'./img/{img_id}.jpg'
        
        question = json.loads(q['Question'])

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # model specific code block
        # ----------------------------------------------------------------------------------------
        # get output from model
        model_answer = handler.ask(input_string=question, img_dir=img_dir, question_type='grounding_o')
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if 'ERROR' in model_answer:
            continue
            
        q['model_answer'] = model_answer
        results.append(q)
        iou_s = compute_iou(true_box, model_answer)
        iou_list.append(iou_s)

        score = sum([1 for x in iou_list if x > 0.3])
        
        pbar.set_postfix({'score': score})
    
    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)

    return score


def test_grounding_t(task, qa_data, model_name, handler, result_dir=None):

    results = []
    iou_list = []
    pnls_list = []
    pbar = tqdm(enumerate(qa_data), desc='grounding_t', total=len(qa_data), ncols=120)
    for i, q in pbar:
        img_id = q['img_id']
        img_dir = f'./img/{img_id}.jpg'
        true_text, true_box = json.loads(q['Answer'])

        question = q['Question']
        
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
        iou_s = compute_iou(true_box, box_a)
        pnls = compute_PNLS(true_text.lower().replace('\"', ''), text_a.lower())

        iou_list.append(iou_s)
        pnls_list.append(pnls)

        score_pnls = sum([1 for x in pnls_list if x > 0.9])
        score_iou = sum([1 for x in iou_list if x > 0.3])

        pbar.set_postfix({'score_pnls': score_pnls, 'score_iou': score_iou})
        
    score = score_pnls + score_iou

    if result_dir is not None:
        json.dump(results, open(os.path.join(result_dir, f'{model_name}_{task}.json'), 'w'), indent=4)

    return score



def test(task, qa_data, model_name, handler, result_dir=None, n_qa=50):
    
    q_type = question_type_zoo[task]
    qa_data = qa_data[:n_qa]

    if q_type == 'recognition_t':
        score = test_recognition(task, qa_data, model_name, handler, result_dir=result_dir)

    elif q_type == 'mcq':
        score = test_mcq(task, qa_data, model_name, handler, result_dir=result_dir)

    elif q_type == 'grounding_o':
        score = test_grounding_o(task, qa_data, model_name, handler, result_dir=result_dir)

    elif q_type == 'grounding_t':
        score = test_grounding_t(task, qa_data, model_name, handler, result_dir=result_dir)


    return score



