from PIL import Image
from io import BytesIO
import base64
from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse

from model import MiniGPT4, Minigpt4Input
from model_install import model_install_minigpt

class Item(BaseModel):
        text: str
        image: str




if __name__ == "__main__":
    print("Installing model dependencies")
    model_install_minigpt()
    
    print("Creating MiniGPT model")
    model = MiniGPT4()

    app = FastAPI(debug=True)

    @app.post('/generate')
    def generate_plan(item: Item):
        
        image = Image.open(BytesIO(base64.b64decode(item.image)))
        model_input = Minigpt4Input(item.text, image)
        text = model.generate(model_input)
        return {'text': text}
    
    @app.get("/name")
    def get_model_name():
        return {"name": model.name}


    uvicorn.run(app, host='0.0.0.0', port=8080)