import fire
import os
import torch
import torch.utils
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

MODEL_PROMPT_DICT = {
    "LLAMA3": {
        "generation_prompt": "<|start_header_id|>assistant<|end_header_id|>",
        "sys_msg": (
            "Below is an instruction that describes a task."
            " Write a response that appropriately completes the request.\n\n"
        ),
        "chat_template": (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>"
            "\n\n' + message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        ),
    },
    "QwQ-32B": {
        "generation_prompt": "<think>\n",
        "sys_msg": "",
        "chat_template": ""
    },
    "Qwen": {
        "generation_prompt": "<|im_start|>assistant",
        "sys_msg": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "chat_template": ""
    }
}


def get_prompt_info(model_name="LLAMA3"):
    generation_prompt = MODEL_PROMPT_DICT[model_name]["generation_prompt"]
    sys_msg = MODEL_PROMPT_DICT[model_name]["sys_msg"]
    chat_template = MODEL_PROMPT_DICT[model_name]["chat_template"]
    return generation_prompt, sys_msg, chat_template


def get_mask(p):
    return torch.sigmoid(p)


def compose_new_params(param_name, decomposed_params, learnable_params):
    mm = get_mask(learnable_params[param_name])
    return (
        decomposed_params[f"{param_name}.U"]
        @ torch.diag_embed(decomposed_params[f"{param_name}.S"] * mm)
        @ decomposed_params[f"{param_name}.V"].T
    ) * (
        decomposed_params[f"{param_name}.S"].sum()
        / (decomposed_params[f"{param_name}.S"] * mm).sum()
    )


@torch.no_grad()
def forward(model, base_params, decomposed_params, learnable_params):
    """Forward pass."""
    new_params = {}
    for k in base_params:
        if "layers.0.mlp.gate_proj" in k:
            new_params[k] = compose_new_params(k, decomposed_params, learnable_params)
            model.get_parameter(k).copy_(new_params[k])
        else:
            new_params[k] = base_params[k]
    return new_params


def backward(model, base_params, decomposed_params, learnable_params):
    """Backward pass."""
    for k in base_params:
        if "layers.0.mlp.gate_proj" in k:
            compose_new_params(k, decomposed_params, learnable_params).backward(
                model.get_parameter(k).grad
            )


def prepare_model_input(tokenizer, train_data, idx, generation_prompt):
    """Return input_ids and label of batch"""
    text = train_data["text"][idx]
    input_ids = tokenizer.encode(text)
    response_template_ids = tokenizer.encode(
        generation_prompt, add_special_tokens=False
    )
    input_ids = torch.tensor(input_ids)
    response_template_ids = torch.tensor(response_template_ids)

    # Find where the full template sequence starts
    template_start = None
    for i in range(len(input_ids) - len(response_template_ids) + 1):
        if torch.all(
            input_ids[i : i + len(response_template_ids)] == response_template_ids
        ):
            template_start = i + len(response_template_ids)
            break

    labels = torch.full_like(input_ids, -100)
    labels[template_start:] = input_ids[template_start:]
    return input_ids, labels


def get_dataset(
    tokenizer,
    samples,
    ixs=None,
    sys_msg="",
    chat_template="",
    input_filed="src",
    output_field="tgt",
    model_name=""
) -> Dataset:
    context_msg = {"role": "system", "content": sys_msg}
    ixs = range(len(samples))
    lines = []
    for ix in ixs:
        if model_name == "LLAMA3":
            user_msg = {"role": "user", "content": samples[input_filed][ix]}
            prompt = tokenizer.apply_chat_template(
                conversation=[context_msg, user_msg],
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=True,
            )
            answer = samples[output_field][ix]
        elif model_name == "QwQ-32B":
            messages = [{"role": "user", "content": samples[input_filed][ix]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = "\n</think>\n\n" + samples[output_field][ix]
        elif model_name == "Qwen":
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": samples[input_filed][ix]}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            answer = samples[output_field][ix]
        
        lines.append(prompt + answer)

    print(f"data sample: \n{lines[0]}")
    dataset = Dataset.from_dict({"text": lines})
    return dataset


def main(
    model_name: str = "LLAMA3",
    model_id="/data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct",
    decomposed_param_file="/data/ldn/self-adaptive-llms/medical/llama3_decomposed_params.pt",
    device="cuda",
    dataset_dir: str = "",
    dataset_name: str = "骨科康复",
    num_iters: int = 50,
    lr: float = 2e-3,
    batch_size: int = 8,
    seed: int = 42,
    init_val: float = 0.1,
    use_wandb: bool = False,
):
    # log related
    if use_wandb:
        import wandb

        _ = wandb.init(
            project="proj-ayu-v2",
            name=f"{dataset_name}-init_val_{init_val}-lr_{lr}-bs_{batch_size}",
            config={
                "lr": lr,
                "seed": seed,
                "batch_size": batch_size,
                "custom_prefix": dataset_name,
                "init_val": init_val,
            },
        )
    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d-%H%M%S")
    log_dir = f"results/medical/{dataset_name}/{model_name}-{datetime_str}"
    os.makedirs(log_dir, exist_ok=True)

    # get model prompt template
    generation_prompt, sys_msg, chat_template = get_prompt_info(model_name=model_name)

    # load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    json_file = os.path.join(dataset_dir, dataset_name + ".json")
    train_val_data = load_dataset("json", data_files=json_file, split="train")
    train_val_data = get_dataset(
        tokenizer, 
        train_val_data, 
        sys_msg=sys_msg, 
        chat_template=chat_template,
        model_name=model_name
    )
    train_size = len(train_val_data)
    num_iters = train_size // batch_size
    print(f"num iters is {num_iters}")
    print(f"train_size: {train_size}")
    train_ix = range(
        0,
        train_size,
    )
    np_random = np.random.RandomState(seed)

    # model and parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # load model parameters
    base_params = model.state_dict()
    # Load decomposed parameters
    if not os.path.exists(decomposed_param_file):
        print("Decomposed params not found. Decomposing...")
        decomposed_params = {}
        for k, v in base_params.items():
            if "layers.0.mlp.gate_proj" in k:
                print(k)
                U, S, V = torch.svd(v.to(torch.float32))
                decomposed_params[f"{k}.U"] = U
                decomposed_params[f"{k}.S"] = S
                decomposed_params[f"{k}.V"] = V
        torch.save(decomposed_params, decomposed_param_file)
    else:
        print("Decomposed params found. Loading...")
        decomposed_params = torch.load(decomposed_param_file)
    for k, v in decomposed_params.items():
        decomposed_params[k] = v.to(torch.bfloat16).to(device)

    # Create learnable parameters. layers.0.mlp.gate_proj u s v
    learnable_params = {}
    num_params = 0
    for k, v in base_params.items():
        if "layers.0.mlp.gate_proj" in k:
            learnable_params[k] = torch.nn.Parameter(
                data=(
                    torch.randn(
                        min(v.shape),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    * 0.01
                    + init_val
                ),
                requires_grad=True,
            )
            num_params += learnable_params[k].numel()
    print(f"#params={num_params}")
    learnable_params_list = list(learnable_params.values())
    optimizer = torch.optim.Adam(learnable_params_list, lr=lr)

    model.eval()
    for k in learnable_params:
        model.get_parameter(k).requires_grad_(True)

    # Training loop.
    for i in range(num_iters):
        batch_ix = np_random.choice(train_ix, size=batch_size, replace=False)

        # fetch data for this batch
        batch_inputs = [
            prepare_model_input(
                tokenizer, train_val_data, i, generation_prompt=generation_prompt
            )
            for i in batch_ix
        ]

        # Update model params
        forward(model, base_params, decomposed_params, learnable_params)

        step_loss = 0.0
        for batch_input in batch_inputs:
            input_ids, labels = batch_input
            input_ids = input_ids.unsqueeze(0).to(model.device)
            labels = labels.unsqueeze(0).to(model.device)

            # check the model input format match
            loss = model(input_ids=input_ids, labels=labels).loss
            print(f"loss is: {loss}")
            step_loss += loss.item()
            loss.backward()

        # Backpropogate and update.
        backward(model, base_params, decomposed_params, learnable_params)

        grad_mean = learnable_params_list[0].grad.mean().item()
        torch.nn.utils.clip_grad_norm_(learnable_params_list, 1.0)  # use default value
        grad_norm_mean = learnable_params_list[0].grad.mean().item()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        # Inaccurate logging.
        print(
            f"Iter {i}, loss = {step_loss / batch_size} "
            + f"param={learnable_params_list[0].mean()}, "
            + f"grad={grad_mean}"
            + f"grad_norm_mean={grad_norm_mean}"
        )

    # 结束保存可训练参数
    # forward(model, base_params, decomposed_params, learnable_params)
    learnable_params_cpu = {
        k: v.cpu() if torch.is_tensor(v) else v for k, v in learnable_params.items()
    }

    # Save the dictionary with CPU tensors
    torch.save(learnable_params_cpu, f"{log_dir}/learnable_params_latest.pt")
    print(f"param saved in {log_dir}/learnable_params_latest.pt")


if __name__ == "__main__":
    fire.Fire(main)
