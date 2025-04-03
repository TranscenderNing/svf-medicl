from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/dev/shm/models1/Qwen2.5-72B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="bfloat16",
#     device_map="auto"
# )
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print('text',text)

# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print('response',response)
