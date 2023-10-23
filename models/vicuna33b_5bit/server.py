from pydantic import BaseModel
import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from model import Vicuna33B



if __name__ == "__main__":
    model = Vicuna33B()
    
    class Item(BaseModel):
        prompt: str

    app = FastAPI(debug=True)

    @app.post('/generate')
    def generate_plan(item: Item):
        output = model.generate(item.prompt)
        return {'output': output}


    uvicorn.run(app, host='0.0.0.0', port=8080)