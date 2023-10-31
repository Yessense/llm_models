from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from model import Vicuna13B


class Item(BaseModel):
        text: str
        image: str


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--num_devices', type=int, default=2)
    args = parser.parse_args()

    model = Vicuna13B(
        num_devices=args.num_devices,
        max_new_tokens=args.max_new_tokens
    )

    app = FastAPI(debug=True)

    @app.post('/generate')
    def generate_plan(item: Item):
        text = model.generate(item.text)
        return {'text': text}

    @app.get("/name")
    def get_model_name():
        return {"name": model.name}
    
    uvicorn.run(app, host='0.0.0.0', port=8080)