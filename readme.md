# Docker containers for LLMs

## Models
There are several aviliable models to run
1. vicuna7b
2. vicuna13b
3. vicuna33b_5bit (quantized to 5bit)

To build docker image with specific model one need to run script
```bash
./build.sh --model model_name
```
To run container with model one need to run almoust the same script
```bash
./run.sh --model model_name
```
