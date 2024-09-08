#!/bin/bash

# Arguments instruction:
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/llava-v1_1-7b", path to a directory where LLaVA hugginface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L hugginface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/gsva-7b-ft-gres.bin", path to a pretrained GSVA checkpoint if finetune.
# --precision="bf16", precision for training.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# num_classes_per_sample=5, 5 is one of the optimum values of how many classes / objects sampled in one training example

export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

ds --master_port=24989 main.py \
  --segmentation_model_path=<path-to-sam-model> \
  --mllm_model_path=<path-to-llava-model> \
  --vision-tower=<path-to-clip-model> \
  --dataset_dir=<path-to-datasets> \
  --weight=<path-to-model-weights> \
  --precision=<precision> \
  --lora_r=<model_lora_rank> \
  --num_classes_per_sample=<some-value>