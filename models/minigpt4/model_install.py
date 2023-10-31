import argparse 
import sys
import json
import pathlib
import os
import gdown
import yaml
from pathlib import Path


MOUNTED_FOULDER = "/root/.cache/huggingface/hub"


def model_install_minigpt():
    # -------------------- Saving paths --------------------
    default_cache_dir = MOUNTED_FOULDER
    vicuna_space = "Vision-CAIR"
    vicuna_id = "vicuna"
    vicuna_repo_id = f"{vicuna_space}/{vicuna_id}"
    chkpt_foulder = f'{default_cache_dir}/pretrained_minigpt4.pth'
    
    # ---------------- Downloading models ----------------
    if not Path(f"{default_cache_dir}/MiniGPT-4").is_dir():
        print(f"[INFO] Clonning Minigpt4 repo in {default_cache_dir}/MiniGPT-4")
        os.system(f"git clone https://github.com/Vision-CAIR/MiniGPT-4.git {default_cache_dir}/MiniGPT-4")
    else:
        print(f"[INFO] NOT clonning Minigpt4 repo in {default_cache_dir}/MiniGPT-4. Foulder already exist")
        
    if not Path(f"{default_cache_dir}/models--{vicuna_space}--{vicuna_id}").is_dir():
        print(f"[INFO] Clonning Minigpt4 repo in {default_cache_dir}/models--{vicuna_space}--{vicuna_id}")
        os.system(f"git lfs clone https://huggingface.co/{vicuna_repo_id} {default_cache_dir}/models--{vicuna_space}--{vicuna_id}")
    else:
        print(f"[INFO] NOT clonning Minigpt4 repo in {default_cache_dir}/models--{vicuna_space}--{vicuna_id}. Foulder already exist")
        
    if not Path(chkpt_foulder).is_file():
        print(f"[INFO] Clonning Minigpt4 repo in {chkpt_foulder}")
        Path(chkpt_foulder).mkdir(parents=True)
        gdown.download("https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link", chkpt_foulder, fuzzy=True)
    else:
        print(f"[INFO] NOT clonning Minigpt4 repo in {chkpt_foulder}. Foulder already exist")
    
    
    # ---------------- Changing model chkpts paths in config ----------------
    for space, repo in [(vicuna_space, vicuna_id)]:
        for path in pathlib.Path(f"{default_cache_dir}/models--{space}--{repo}/snapshots/").rglob("*/tokenizer_config.json"):
            print(f"Loading {path}")
            config = json.loads(open(path, "r").read())
            if config["tokenizer_class"] == "LlamaTokenizer":
                print("No fix needed")
            else:   
                config["tokenizer_class"] = "LlamaTokenizer"
            with open(path, "w") as f:
                json.dump(config, f)
    
    eval_config_path = pathlib.Path(f"{default_cache_dir}/MiniGPT-4/eval_configs/minigpt4_eval.yaml")
    with open(eval_config_path, "r") as f:
        eval_config_dict = yaml.safe_load(f)
        eval_config_dict["model"]["ckpt"] = f"{default_cache_dir}/pretrained_minigpt4.pth"
        eval_config_dict["model"]["prompt_path"] = f"{default_cache_dir}/MiniGPT-4/prompts/alignment.txt"
        eval_config_dict["model"]["low_resource "] = False
        
    with open(eval_config_path, "w") as f:
        yaml.dump(eval_config_dict, f)

    minigpt4_config_path = pathlib.Path(f"{default_cache_dir}/MiniGPT-4/minigpt4/configs/models/minigpt4_vicuna0.yaml")
    with open(minigpt4_config_path, "r") as f:
        minigpt4_config_dict = yaml.safe_load(f)
        minigpt4_config_dict["model"]["llama_model"] = f"{default_cache_dir}/models--{vicuna_space}--{vicuna_id}"
        
    with open(minigpt4_config_path, "w") as f:
        yaml.dump(minigpt4_config_dict, f)
    
    
     # ---------------- Installing sub models ----------------
    # minigpt4_path = f'{default_cache_dir}/MiniGPT-4'
        
    # if sys.path[-1] != minigpt4_path:
    #     sys.path.append(minigpt4_path)

    # from minigpt4.common.config import Config
    # from minigpt4.common.registry import registry
    # from minigpt4.conversation.conversation import StoppingCriteriaSub, Conversation, SeparatorStyle

    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--cfg-path', help='')
    # parser.add_argument('--options', nargs="+",help='')
    # parser.add_argument('--gpu-id', default=0, help='')
    # args = parser.parse_args(f" --cfg-path {default_cache_dir}/MiniGPT-4/eval_configs/minigpt4_eval.yaml".split())

    # cfg = Config(args)

    # model_config = cfg.model_cfg
    # model_config.device_8bit = args.gpu_id
    # model_cls = registry.get_model_class(model_config.arch)

    # vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train

    
    # print("<MODEL CLS FROM CONFIG>")
    # _model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    # _model.eval()
    # print("</MODEL CLS FROM CONFIG>")
    # print("<MODEL VIS FROM CONFIG>")
    # _vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    # print("</MODEL VIS FROM CONFIG>")


if __name__ == "__main__":
    model_install_minigpt()