import json


def calculate_accuracy(file_path):
    # 读取 JSON 文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    correct_count = 0
    total_count = len(data)

    # 遍历所有数据并比较 'tgt' 和 'model_output'
    for item in data:
        if len(item['tgt']) == 1 and item['tgt'] == item['model_output'][0]:
            correct_count += 1
        elif item['model_output'].replace('/', '') == item['tgt']:
            correct_count += 1

    print(f"Total count/correct count: {total_count} / {correct_count}")
    # 计算准确率
    accuracy = correct_count / total_count * 100
    return accuracy

# 微调后的模型
file_path = '/home/ldn/baidu/reft-pytorch-codes/svf-medicl/results/results_5domain_72b.json'
accuracy = calculate_accuracy(file_path)
print(f'准确率: {accuracy:.2f}%')


# 基座模型，没有经过微调
file_path = '/home/ldn/baidu/reft-pytorch-codes/svf-medicl/results/results_5domain_72b_without_finetune.json'
accuracy = calculate_accuracy(file_path)
print(f'准确率: {accuracy:.2f}%')

