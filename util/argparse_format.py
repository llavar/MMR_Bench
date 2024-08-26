import argparse
from typing import List


task_description = ('font_color:        MCQ of font color\n'
                    'font_size:         MCQ of comparing font size\n'
                    'localization_o:    MCQ of object position\n'
                    'localization_t:    MCQ of text position\n'
                    'spatial_OO:        MCQ of object to object spatial relation\n'
                    'spatial_OT:        MCQ of object to text spatial relation\n'
                    'spatial_TT:        MCQ of text to text spatial relation\n'
                    'recognition_label: Text Recognization given label\n'
                    'recognition_pos:   Text Recognization given label\n'
                    'grounding_o:       Bounding box detection\n'
                    'grounding_t:       Text Recognization + Bounding box detection\n'
                   )

model_zoo = ['GPT-4V', 'GPT-4o', 'Claude-3.5-Sonnet', 'Phi-3-Vision', 'Qwen-vl-plus', 'Qwen-vl-max', 'LLaVA-NEXT', 'LLaVA-NEXT-34B',
             'LLaVA-1.5', 'Monkey', 'idefics-80b', 'idefics-2-8b', 'chameleon-7b', 'chameleon-30b', 'InternVL2-1B', 'InternVL2-2B', 
             'InternVL2-4B', 'InternVL2-8B', 'InternVL2-76B']


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text: str, width: int) -> List[str]:
        lines: List[str] = []
        for line_str in text.split('\n'):
            line: List[str] = []
            line_len = 0
            for word in line_str.split(' '):
                word_len = len(word)
                next_len = line_len + word_len
                if line: 
                    next_len += 1
                if next_len > width:
                    lines.append(' '.join(line))
                    line.clear()
                    line_len = 0
                if line_len:
                    line_len += 1
                line.append(word)
                line_len += word_len
            if line:
                lines.append(' '.join(line))
        return lines

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return '\n'.join(indent + line for line in self._split_lines(text, width - len(indent)))



parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
parser.add_argument("--task", default='font_color', help=f"Name of task to test, choose from following:\n{task_description}\n", type=str)
parser.add_argument("--model", default=None, help=f"Model to test, choose from:\n{model_zoo}\n", type=str)
parser.add_argument("--result_dir", default=None, help="result_dir to save results", type=str)
parser.add_argument("--device", default='cuda', help="Device to use (cuda/cpu), default is auto choice that prefer cuda if available", type=str)
MMR_args = parser.parse_args()






