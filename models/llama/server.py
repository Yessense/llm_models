from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from model import LLaMA


class Item(BaseModel):
        text: str
        image: str


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str,choices = ["7B", "13B"], default="7B")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model = LLaMA(device=args.device,
                  model_size=args.model_size)

    app = FastAPI(debug=True)

    @app.post('/generate')
    def generate_plan(item: Item):
        output = model.generate(item.text)
        return {'text': output}

    @app.get("/name")
    def get_model_name():
        return {"name": model.name}

    uvicorn.run(app, host='0.0.0.0', port=8080)