from PIL import Image
from typing import Optional
from dataclasses import dataclass
import argparse 

import sys
MOUNTED_FOULDER = "/root/.cache/huggingface/hub"
minigpt4_path = f'{MOUNTED_FOULDER}/MiniGPT-4'
if sys.path[-1] != minigpt4_path:
    sys.path.append(minigpt4_path)
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from MiniGPT4Chat import MiniGPT4Chat


@dataclass
class Minigpt4Input:
    text: Optional[str] = None
    image: Optional[Image.Image] = None


class MiniGPT4:
    def __init__(self,
                 device: int = 2,
                 name: str = 'minigpt4',
                 max_new_tokens: int = 200) -> None:
        self._num_beams = 1
        self._temperature = 0.9
        self._max_new_tokens = max_new_tokens
        self.name = name
        self._load()

    def _load(self):
        # ---------------- Creating model class ----------------
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--cfg-path', help='')
        parser.add_argument('--options', nargs="+",help='')
        parser.add_argument('--gpu-id', default=0, help='')
        args = parser.parse_args(f" --cfg-path {MOUNTED_FOULDER}/MiniGPT-4/eval_configs/minigpt4_eval.yaml".split())

        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train

        # ---------------- Creating model instance ----------------
        self._model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        self._model.eval()
        self._vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.minigpt4 = MiniGPT4Chat(self._model, self._vis_processor)


    def generate(self, inputs: Minigpt4Input) -> str:
        self.minigpt4.reset_history()
        
        if inputs.image is not None:
            self.minigpt4.upload_img(inputs.image)
        self.minigpt4.ask(inputs.text)
        out, _ = self.minigpt4.answer(
            num_beams=self._num_beams,
            temperature=self._temperature,
            max_new_tokens=self._max_new_tokens,
        )    
        return out
            
        