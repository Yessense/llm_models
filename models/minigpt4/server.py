# answer on all questions 
# https://stackoverflow.com/questions/6485790/numpy-array-to-base64-and-back-to-numpy-array-python

from PIL import Image
from io import BytesIO
import base64
from PIL import Image
# import numpy as np
import pickle
import codecs
from pydantic import BaseModel
import uvicorn
import requests
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
import gc
from model_install import model_install_minigpt
import torch
print("Installing model dependencies")
model_install_minigpt()

from model import MiniGPT4, Minigpt4Input
from typing import Optional, Any



def str2obj(raw_data: str):
    if raw_data is None:
        return None
    object = pickle.loads(codecs.decode(raw_data.encode('latin1'), "base64"))
    return object


def obj2str(obj: Any):
    if obj is None:
        return None
    raw_data = codecs.encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), "base64").decode('latin1')
    return raw_data


class GenerationInput(BaseModel):
        promt: Optional[str] = None
        image: Optional[str] = None
        images: Optional[dict] = None


class SubtaskQueueInput(BaseModel):
        goal: Optional[str] = None


class HelpForNavInput(BaseModel):
        subtask: Optional[str] = None
        image: Optional[str] = None


class ReplanInput(BaseModel):
        error: Optional[str] = None
        image: Optional[str] = None
        classes: Optional[str] = None
        cur_subtask: Optional[str] = None


if __name__ == "__main__":
    print("Creating MiniGPT model")
    model = MiniGPT4()

    app = FastAPI(debug=True)

    # @app.post('/get_subtask_queue')
    # def get_subtask_queue(item: SubtaskQueueInput):
    #     print(f"Recieving at 'get_subtask_queue' {item=}")
        
    #     model_input = Minigpt4Input(item.goal)
    #     model_output = model.generate(model_input)
        
    #     print(f"Model has predicted {model_output}")
    #     return {'text': model_output}

    # @app.post('/get_help_for_nav')
    # def get_help_for_nav(item: HelpForNavInput):
    #     print(f"Recieving at 'get_help_for_nav' {item=}")
        
    #     image = str2obj(item.image)
    #     model_input = Minigpt4Input(item.goal, image)
    #     model_output = model.generate(model_input)
        
    #     print(f"Model has predicted {model_output}")
    #     return {'text': model_output}
    
    # @app.post('/replan_subtasks')
    # def replan_subtasks(item: ReplanInput):
    #     print(f"Recieving at 'replan_subtasks' {item=}")
        
    #     image = str2obj(item.image)
    #     text = item.error + str(item.classes)
    #     model_input = Minigpt4Input(text, image)
    #     model_output = model.generate(model_input)
        
    #     print(f"Model has predicted {model_output}")
    #     return {'text': model_output}
    
    @app.post('/generate')
    def generate_plan(item: GenerationInput):
        gc.collect()
        torch.cuda.empty_cache()

        promt = item.promt
        image = str2obj(item.image)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        images = None
        print(f"Model input '{promt}' and Image={image is not None}, Images_len={len(item.images) if item.images is not None else 'None'}")
        if item.images is not None:
            images = [str2obj(img) for k, img in item.images.items()]
            images += [image]
            image = None
        
        model_input = Minigpt4Input(promt, image, images)
        text = model.generate(model_input)

        gc.collect()
        torch.cuda.empty_cache()
        return {'text': text}

    @app.get("/name")
    def get_model_name():
        return {"name": model.name}


    uvicorn.run(app, host='0.0.0.0', port=8080)
