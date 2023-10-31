from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from model import Vicuna33B


class Item(BaseModel):
        text: str
        image: str


if __name__ == "__main__":
    model = Vicuna33B()

    app = FastAPI(debug=True)

    @app.post('/generate')
    def generate_plan(item: Item):
        text = model.generate(item.text)
        return {'text': text}

    @app.get("/name")
    def get_model_name():
        return {"name": model.name}
    
    uvicorn.run(app, host='0.0.0.0', port=8080)