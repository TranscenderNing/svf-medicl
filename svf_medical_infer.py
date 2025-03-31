import fire
import os
import torch
import torch.utils
from datasets import load_dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

LEARNABLE_PARAM_PATHS = {
    "卒中康复": "/data/ldn/svf-medicl/results/medical/卒中康复/20250327-163844/learnable_params_latest.pt",
    "脊髓损伤康复": "/data/ldn/svf-medicl/results/medical/脊髓损伤康复/20250327-170755/learnable_params_latest.pt",
    "内科康复": "/data/ldn/svf-medicl/results/medical/内科康复/20250327-172659/learnable_params_latest.pt",
    "脊柱康复": "/data/ldn/svf-medicl/results/medical/脊柱康复/20250327-172815/learnable_params_latest.pt",
    "四肢康复": "/data/ldn/svf-medicl/results/medical/四肢康复/20250327-173343/learnable_params_latest.pt",
}


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
        # target_modules=['self_attn.q_proj', 'self_attn.v_proj']
        if "mlp" in k:
            new_params[k] = compose_new_params(k, decomposed_params, learnable_params)
            model.get_parameter(k).copy_(new_params[k])
        else:
            new_params[k] = base_params[k]
    return new_params


def backward(model, base_params, decomposed_params, learnable_params):
    """Backward pass."""
    for k in base_params:
        if "mlp" in k:
            compose_new_params(k, decomposed_params, learnable_params).backward(
                model.get_parameter(k).grad
            )


def compute_metric(
    model,
    tokenizer,
    base_params,
    decomposed_params,
    device=None,
    test_dir="",
    datasets=[],
    batch_size=8,
    input_field="",
    label_field="",
    output_file="",
    max_new_tokens=10,
):
    all_ds = {}

    for dataset_name in datasets:
        json_file = os.path.join(test_dir, dataset_name + ".json")
        test_ds = load_dataset(path="json", data_files=json_file, split="train")
        all_ds[dataset_name] = test_ds

    total_size = sum(
        len(value) if hasattr(value, "__len__") else 0 for value in all_ds.values()
    )

    def modify_question(example):  # -> Any:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example[input_field]},
        ]
        example[input_field] = tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )
        return example

    correct_count = 0
    results = []
    for domain, path in LEARNABLE_PARAM_PATHS.items():
        print(f"Testing {domain}...")
        # load model
        learnable_params = torch.load(path, map_location=device)
        print("Learnable params loaded.")
        forward(model, base_params, decomposed_params, learnable_params)
        print("model loaded")
        # load data
        test_ds = all_ds[domain]
        test_ds = test_ds.map(modify_question)
        batched_test_ds = test_ds.batch(batch_size)
        print("anwser and generation")
        for batch in batched_test_ds:
            model_inputs = tokenizer(
                batch[input_field], return_tensors="pt", padding="longest"
            ).to(device)
            generated_ids = model.generate(
                **model_inputs, max_new_tokens=max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for question, label, generation in zip(
                batch[input_field], batch[label_field], response
            ):
                print(f"label is {label},  model output is {generation}")
                if label in generation:
                    correct_count += 1
                results += [
                    {
                        input_field: question,
                        label_field: label,
                        "model_output": generation,
                        "cls": domain,
                    }
                ]
    print(f"Accuracy: {correct_count / total_size}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return


def main(
    model_id="/data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct",
    decomposed_param_file="/data/ldn/self-adaptive-llms/medical/llama3_decomposed_params.pt",
    device="cuda",
    test_dir="/data/ldn/svf-medicl/data/test",
    datasets=["卒中康复","脊髓损伤康复","内科康复", "脊柱康复四肢康复"],
    input_field="src",
    label_field="tgt",
    output_file="results/results_5domain.json",
    max_new_tokens=10,
    batch_size: int = 8,
    init_val: float = 0.1,
):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # load model and parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    base_params = model.state_dict()
    # Load decomposed parameters.
    if not os.path.exists(decomposed_param_file):
        print("Decomposed params not found. Decomposing...")
        decomposed_params = {}
        for k, v in base_params.items():
            if "mlp" in k:
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


    # inference and test
    model.eval()
    compute_metric(
        model,
        tokenizer,
        base_params,
        decomposed_params,
        device=device,
        test_dir=test_dir,
        datasets=datasets,
        batch_size=batch_size,
        input_field=input_field,
        label_field=label_field,
        output_file=output_file,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    fire.Fire(main)
