from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from model import Vicuna
from typing import Optional


class Item(BaseModel):
        prompt: Optional[str] = None
        image: Optional[str] = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str,choices = ["7B", "13B"], default="7B")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model = Vicuna(device=args.device,
                  model_size=args.model_size)

    app = FastAPI(debug=True)

    @app.post('/generate')
    def generate_plan(item: Item):
        print(f"-------------LLM INPUT-------------\n{item.prompt}")
        text = model.generate(item.prompt)
        print(f"-------------LLM OUTPUT-------------\n{text}")
        return {'text': text}

    @app.get("/name")
    def get_model_name():
        return {"name": model.name}
    
    uvicorn.run(app, host='0.0.0.0', port=8080)