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

model_zoo_list = ', '.join(['GPT-4V', 'GPT-4o', 'Phi-3-Vision', 'Qwen-vl-plus', 'Qwen-vl-max', 'LLaVA-NEXT',
              'LLaVA-1.5', 'Monkey'])

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
parser.add_argument("--model", default='GPT-4V', help=f"Model to test, choose from:\n{model_zoo_list}\n", type=str)
# parser.add_argument("--pre_prompt", default=None, help="Prompt before the question", type=str)
# parser.add_argument("--post_prompt", default=None, help="Prompt after the question", type=str)
# parser.add_argument("--system_prompt", default='None', help="System prompt required for gpt4v and gpto", type=str)
parser.add_argument("--device", default='auto', help="Device to use (cuda/cpu), default is auto choice that prefer cuda if available", type=str)
MMR_args = parser.parse_args()






