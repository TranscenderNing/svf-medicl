

#!/bin/bash

# 定义数据集数组
# ['卒中康复.jsonl', '脊髓损伤康复.jsonl', '内科康复.jsonl', '脊柱康复.jsonl', '四肢康复.jsonl']
datasets=('脊髓损伤康复' '内科康复' '脊柱康复' '四肢康复')

# 遍历数据集并运行 Python 脚本
for dataset in "${datasets[@]}"
do
  # 打印当前正在处理的数据集名称
  echo "Processing dataset: $dataset"
  
  # 运行 Python 脚本
  CUDA_VISIBLE_DEVICES=1 python svf_medical_train.py \
  --model_name Qwen \
  --model_id /dev/shm/models1/Qwen2.5-72B-Instruct \
  --decomposed_param_file Qwen2.5-72B-Instruct_decomposed_params.pt \
  --device cuda \
  --dataset_dir data/train \
  --dataset_name $dataset \
  --num_iters 50 \
  --lr 2e-3 \
  --batch_size 2 \
  --seed 42 \
  --init_val 0.1 \
  --use_wandb True
  
done




