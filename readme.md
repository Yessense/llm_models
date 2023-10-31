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

You can run model on specific gpu by passing "--gpu_id" in run.sh
```bash
./run.sh --model minigpt4 --gpu_id "device=1"    # running model on cuda:1
./run.sh --model minigpt4 --gpu_id "device=0,1"  # running model on cuda:0 and 1
./run.sh --model minigpt4 --gpu_id "all"         # running model on all avilable gpus
./run.sh --model minigpt4                        # by default --gpu_id is set to "all"
```