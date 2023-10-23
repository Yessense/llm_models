#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) model="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

case $model in 
    vicuna7b)
    echo "Running container for model Vicuna7b"
    docker run --rm -it --gpus all -p 8080:8080 -v "$HOME"/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ --name llm_models.vicuna7b vicuna7b:latest 
    ;;

    vicuna13b)
    echo "Running container for model Vicuna13b"
    docker run --rm -it --gpus all -p 8080:8080 -v "$HOME"/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ --name llm_models.vicuna13b vicuna13b:latest 
    ;;

    vicuna33b_5bit)
    echo "Running container for model Vicuna33b-5bit"
    docker run --rm -it --gpus all -p 8080:8080 -v "$HOME"/.cache/huggingface/hub/:/root/.cache/huggingface/hub/ --name llm_models.vicuna33b_5bit vicuna13b:latest 
    ;;

    llama2)
    echo "Not implemented yet"
    ;;

    *)
    echo "No existing docker found for model $model"
    echo "Interrupting process"

    ;;
esac


