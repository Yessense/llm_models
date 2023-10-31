#!/bin/bash

unified_build(){
    model_name=$1
    echo "Building image for model $model"
    docker build -f models/$model/dockerfile -t $model:latest .
}


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) model="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


case $model in 
    vicuna7b)
    unified_build $model
    ;;

    vicuna13b)
    unified_build $model
    ;;
    
    vicuna33b_5bit)
    unified_build $model
    ;;

    llama2)
    unified_build $model
    ;;

    minigpt4)
    unified_build $model
    ;;

    *)
    echo "No existing docker found for model $model"
    echo "Interrupting process"

    ;;
esac


