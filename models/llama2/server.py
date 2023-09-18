from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from model import LLaMA2



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str,choices = ["7B", "13B"], default="7B")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model = LLaMA2(device=args.device,
                  model_size=args.model_size)

    class Item(BaseModel):
        prompt: str

    app = FastAPI(debug=True)

    @app.post('/generate')
    def generate_plan(item: Item):
        output = model.generate(item.prompt)
        return {'output': output}


    uvicorn.run(app, host='0.0.0.0', port=8080)