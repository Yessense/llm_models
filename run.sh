#!/bin/bash

unified_run(){
    model_name=$1
    gpu_id=$2
    echo "Running container for model $model"
    docker run --rm -it --gpus $gpu_id -p 8080:8080 -v "$HOME"/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ --name llm_models.$model $model:latest
}

gpu_id="all"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) model="$2"; shift ;;
        --gpu_id) gpu_id="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


case $model in 
    vicuna7b)
    unified_run $model $gpu_id
    ;;

    vicuna13b)
    unified_run $model $gpu_id
    ;;

    vicuna33b_5bit)
    unified_run $model $gpu_id
    ;;

    minigpt4)
    unified_run $model $gpu_id
    ;;

    llama2)
    echo "Not implemented yet"
    ;;

    *)
    echo "No existing docker found for model $model"
    echo "Interrupting process"
    ;;
esac


