from PIL import Image
from io import BytesIO
import base64
from pydantic import BaseModel
import uvicorn
import requests
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse

from model import MiniGPT4, Minigpt4Input
from model_install import model_install_minigpt

from typing import Optional


def list2np(obj: list[list]) -> Optional[np.ndarray]:
    if obj is None:
        return None
    
    res = np.array(obj)
    return res

class GenerationInput(BaseModel):
        text: Optional[str] = None
        image: Optional[list] = None

class SubtaskQueueInput(BaseModel):
        goal: Optional[str] = None

class HelpForNavInput(BaseModel):
        subtask: Optional[str] = None
        image: Optional[list] = None

class ReplanInput(BaseModel):
        error: Optional[str] = None
        image: Optional[list] = None
        classes: Optional[str] = None
        cur_subtask: Optional[str] = None

if __name__ == "__main__":
    print("Installing model dependencies")
    model_install_minigpt()
    print("Creating MiniGPT model")
    model = MiniGPT4()

    app = FastAPI(debug=True)

    @app.post('/get_subtask_queue')
    def get_subtask_queue(item: SubtaskQueueInput):
        print(f"Recieving at 'get_subtask_queue' {item=}")
        
        model_input = Minigpt4Input(item.goal)
        model_output = model.generate(model_input)
        
        print(f"Model has predicted {model_output}")
        return {'text': model_output}

    @app.post('/get_help_for_nav')
    def get_help_for_nav(item: HelpForNavInput):
        print(f"Recieving at 'get_help_for_nav' {item=}")
        
        image = list2np(item.image)
        model_input = Minigpt4Input(item.goal, image)
        model_output = model.generate(model_input)
        
        print(f"Model has predicted {model_output}")
        return {'text': model_output}
    
    @app.post('/replan_subtasks')
    def replan_subtasks(item: ReplanInput):
        print(f"Recieving at 'replan_subtasks' {item=}")
        
        image = list2np(item.image)
        text = item.error + str(item.classes)
        model_input = Minigpt4Input(text, image)
        model_output = model.generate(model_input)
        
        print(f"Model has predicted {model_output}")
        return {'text': model_output}
    
    @app.post('/generate')
    def generate_plan(item: GenerationInput):
        image = list2np(item.image)
        model_input = Minigpt4Input(item.text, image)
        model_output = model.generate(model_input)
        return {'text': model_output}
    
    @app.get("/name")
    def get_model_name():
        return {"name": model.name}


    uvicorn.run(app, host='0.0.0.0', port=8080)