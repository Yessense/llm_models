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
    echo "Building image for model Vicuna7b"
    docker build -f models/vicuna7b/dockerfile -t vicuna7b:latest .
    ;;

    vicuna13b)
    echo "Building image for model Vicuna13b"
    docker build -f models/vicuna13b/dockerfile -t vicuna13b:latest .
    ;;
    
    vicuna33b_5bit)
    echo "Building image for model Vicuna33b_5bit"
    docker build -f models/vicuna13b/dockerfile -t vicuna33b_5bit:latest .
    ;;

    llama2)
    echo "Building image for model LLaMA"
    docker build -f models/llama2/dockerfile -t llama2:latest .
    ;;

    *)
    echo "No existing docker found for model $model"
    echo "Interrupting process"

    ;;
esac


