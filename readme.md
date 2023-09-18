To run llama model:
```bash
docker build -f models/llama/dockerfile -t llama:latest .
```

```
docker run --gpus all -it --rm --name akorchemnyi.llama -p 8080:8080 llama:latest  
```

To run llama2 model:
```bash
docker build -f models/llama2/dockerfile -t llama2:latest .
```

```bash
docker run --gpus all -it --rm --name ampiro.llama2 -p 8080:8080 llama2:latest  
```

To run Vicuna model:
```bash
docker build -f models/vicuna/dockerfile -t vicuna:latest .
```

```bash
docker run --gpus all -it --rm --name ampiro.vicuna -p 8080:8080 vicuna:latest  
```