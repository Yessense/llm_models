```bash
docker build -f models/llama/dockerfile -t llama:latest .
```

```
docker run --gpus all -it --rm --name akorchemnyi.llama -p 8080:8080 llama:latest  
```