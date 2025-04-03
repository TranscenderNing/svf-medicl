# 支持的模型
LLAMMA3


# 一键训练脚本
cd path/to/svf-medical

nohup bash train.sh > logs/train_bash.log 2>&1 &

# 单独训练
CUDA_VISIBLE_DEVICES=1 nohup python svf_medical_train.py \
--model_name LLAMA3 \
--model_id /data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct \
--decomposed_param_file llama3_decomposed_params.pt \
--device cuda \
--dataset_dir data/train \
--dataset_name 卒中康复 \
--num_iters 50 \
--lr 2e-3 \
--batch_size 8 \
--seed 42 \
--init_val 0.1 \
--use_wandb True > logs/train.log 2>&1 &




```
# 单独训练
CUDA_VISIBLE_DEVICES=1,2,3,6 nohup python svf_medical_train.py \
--model_name QwQ-32B \
--model_id /home/ldn/models/QwQ-32B \
--decomposed_param_file QwQ-32B_decomposed_params.pt \
--device cuda \
--dataset_dir data/train \
--dataset_name 卒中康复 \
--num_iters 50 \
--lr 2e-3 \
--batch_size 2 \
--seed 42 \
--init_val 0.1 \
--use_wandb True > logs/train_QwQ-32B.log 2>&1 &
```




```
# 单独训练
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup python svf_medical_train.py \
--model_name Qwen \
--model_id /dev/shm/models1/Qwen2.5-72B-Instruct \
--decomposed_param_file Qwen2.5-72B-Instruct_decomposed_params.pt \
--device cuda \
--dataset_dir data/train \
--dataset_name 卒中康复 \
--num_iters 50 \
--lr 2e-3 \
--batch_size 2 \
--seed 42 \
--init_val 0.1 \
--use_wandb True > logs/train_Qwen2.5-72B-Instruct.log 2>&1 &
```



# 推理
```
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 nohup python svf_medical_infer.py \
  --model_id /dev/shm/models1/Qwen2.5-72B-Instruct \
  --decomposed_param_file Qwen2.5-72B-Instruct_decomposed_params.pt \
  --device cuda \
  --test_dir /home/ldn/baidu/reft-pytorch-codes/svf-medicl/data/test \
  --datasets '["卒中康复", "脊髓损伤康复", "内科康复", "脊柱康复", "四肢康复"]' \
  --input_field src \
  --label_field tgt \
  --output_file results/results_5domain_72b_without_finetune.json \
  --max_new_token 10 \
  --batch_size 8 \
  --init_val 0.1 \
  > logs/infer_72b.log 2>&1 &
```




# 使用没有微调的模型进行推理
```
CUDA_VISIBLE_DEVICES=1,2,6,7 nohup python svf_medical_infer_without_finetune.py \
  --model_id /dev/shm/models1/Qwen2.5-72B-Instruct \
  --device cuda \
  --test_dir /home/ldn/baidu/reft-pytorch-codes/svf-medicl/data/test \
  --datasets '["卒中康复", "脊髓损伤康复", "内科康复", "脊柱康复", "四肢康复"]' \
  --input_field src \
  --label_field tgt \
  --output_file results/results_5domain_72b_without_finetune.json \
  --max_new_token 10 \
  --batch_size 8 \
  > logs/infer_72b_without_finetune.log 2>&1 &
```



```
CUDA_VISIBLE_DEVICES=1,2,6,7 nohup python svf_medical_infer.py \
  --model_id /data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct \
  --decomposed_param_file /data/ldn/svf-medicl/llama3_decomposed_params.pt \
  --device cuda \
  --test_dir /data/ldn/svf-medicl/data/test \
  --datasets '["卒中康复", "脊髓损伤康复", "内科康复", "脊柱康复", "四肢康复"]' \
  --input_field src \
  --label_field tgt \
  --output_file results/results_5domain.json \
  --max_new_token 10 \
  --batch_size 8 \
  --init_val 0.1 \
  > logs/infer.log 2>&1 &

```
       
