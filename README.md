# Evaluating Reading Ability of Large Multimodal Models
The evaluation code for the [MMR: Multi-Modal Reading Benchmark](https://llavar.github.io/mmr/).

## The MMR Benchmark
![image](demo.png)

## Image Data 
Images are available on [`huggingface`](https://huggingface.co/datasets/puar-playground/MMR), and can be loaded automatically.
```
MMR_data = load_dataset('puar-playground/MMR', split='test')
```

## Installation
Create a conda venv
```
conda create -n mmr python=3.10 -y
conda activate mmr
```
Then, do pip install
```
pip install -r requirements.txt
```

## Run inference using reported models 
The `inference.py` takes `model` and `task` flags. Run the python script to test selected model on specific task or use `all` to test on all tasks.
```
CUDA_VISIBLE_DEVICES=0 python inference.py --model LLaVA-NEXT --task all
```

## Run inference using a new model
The [`inference_customize.py`](https://github.com/llavar/MMR_Bench/blob/977e60eb3d2735efcc810462940acc839e6ccdeb/inference_customize.py#L21C5-L44C28) give a demo for defining your own model handler. Modify the `load_model` and `ask` functions to run your own model.

## task and model list
Check the help metadata for a full list of options. 
```
usage: inference.py [-h] [--task TASK] [--model MODEL] [--device DEVICE]

optional arguments:
  -h, --help       show this help message and exit
  --task TASK      Name of task to test, choose from following:
                   font_color:        MCQ of font color
                   font_size:         MCQ of comparing font size
                   localization_o:    MCQ of object position
                   localization_t:    MCQ of text position
                   spatial_OO:        MCQ of object to object spatial relation
                   spatial_OT:        MCQ of object to text spatial relation
                   spatial_TT:        MCQ of text to text spatial relation
                   recognition_label: Text Recognization given label
                   recognition_pos:   Text Recognization given label
                   grounding_o:       Bounding box detection
                   grounding_t:       Text Recognization + Bounding box detection
                   
                   
  --model MODEL    Model to test, choose from:
                   GPT-4V, GPT-4o, Phi-3-Vision, Qwen-vl-plus, Qwen-vl-max, LLaVA-NEXT, LLaVA-1.5, Monkey
                   
  --device DEVICE  Device to use (cuda/cpu), default is auto choice that prefer cuda if available
```


## Reference
```
@inproceedings{zhang2024trins,
  title={TRINS: Towards Multimodal Language Models that Can Read},
  author={Zhang, Ruiyi and Zhang, Yanzhe and Chen, Jian and Zhou, Yufan and Gu, Jiuxiang and Chen, Changyou and Sun, Tong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22584--22594},
  year={2024}
}
```
