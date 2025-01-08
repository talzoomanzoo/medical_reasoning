ddx_file_dict = {
    'claude': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-claude-3.5-sonnet-claude-3.5-sonnet-claude-3.5-sonnet.json',
    'gemma': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-gemma-2-9b-gemma-2-9b-gemma-2-9b.json',
    'gpt3': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-gpt-3.5-turbo-gpt-3.5-turbo-gpt-3.5-turbo.json',
    'gpt4t': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-gpt-4-turbo-gpt-4-turbo-gpt-4-turbo.json',
    'gpt4omini': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-gpt-4o-mini-gpt-4o-mini-gpt-4o-mini.json',
    'llama3170b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-llama-3.1-70b-llama-3.1-70b-llama-3.1-70b.json',
    'llama318b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-llama-3.1-8b-llama-3.1-8b-llama-3.1-8b.json',
    'meditron-70b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-meditron-70b-meditron-70b-meditron-70b.json',
    'meerkat-70b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-meerkat-70b-meerkat-70b-meerkat-70b.json',
    'meerkat8b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-meerkat-8b-meerkat-8b-meerkat-8b.json',
    'mistral7b': 'medical_reasoning/results/0102/multicritic-ddxplus-mistral-7b-mistral-7b-mistral-7b.json',
    'qwen215b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-qwen-2-1.5b-qwen-2-1.5b-qwen-2-1.5b.json',
    'tulu-70b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-ddxplus-tulu-70b-tulu-70b-tulu-70b.json'
}

medqa_file_dict = {
    'claude': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-claude-3.5-sonnet-claude-3.5-sonnet-claude-3.5-sonnet.json',
    'gemma': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-gemma-2-9b-gemma-2-9b-gemma-2-9b.json',
    'gpt_3.5_turbo': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-gpt-3.5-turbo-gpt-3.5-turbo-gpt-3.5-turbo.json',
    'gpt4turbo': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-gpt-4-turbo-gpt-4-turbo-gpt-4-turbo.json',
    'gpt4omini': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-gpt-4o-mini-gpt-4o-mini-gpt-4o-mini.json',
    'llama3170b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-llama-3.1-70b-llama-3.1-70b-llama-3.1-70b.json',
    'llama318b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-llama-3.1-8b-llama-3.1-8b-llama-3.1-8b.json',
    'meditron70b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-meditron-70b-meditron-70b-meditron-70b.json',
    'medllama8B': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-medllama-8B-medllama-8B-medllama-8B.json',
    'meerkat-70b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-meerkat-70b-meerkat-70b-meerkat-70b.json',
    'meerkat8b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-meerkat-8b-meerkat-8b-meerkat-8b.json', 
    'mistral7b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-mistral-7b-mistral-7b-mistral-7b.json',
    'qwen215b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-qwen-2-1.5b-qwen-2-1.5b-qwen-2-1.5b.json',
    'tulu-70b': '/home/lune/nas2/Projects/medical/medical_reasoning/results/0102/multicritic-medqa-tulu-70b-tulu-70b-tulu-70b.json'
}

import json
from datasets import load_dataset, DatasetDict

with open('/home/lune/nas2/Projects/medical/medical_reasoning/sj/raw_data/medqa.json', 'r') as f:
    medqa_data = json.load(f)

with open('/home/lune/nas2/Projects/medical/medical_reasoning/sj/raw_data/ddxplus/chosen_rejected_consistency.json', 'r') as f:
    ddx_data = json.load(f)

medqa_index = medqa_data['example_dict']
ddx_index = ddx_data['model_index_map']

## want to verify that the keys in file dict are in index, and that the json files are exists
for model_name, file_path in ddx_file_dict.items():
    if model_name not in ddx_index:
        print(f"모델 이름 '{model_name}'이 ddx_index에 없습니다.")
    else:
        try:
            with open(file_path, 'r') as f:
                pass  # 파일이 존재하는 경우는 무시
        except FileNotFoundError:
            print(f"'{file_path}' 파일이 존재하지 않습니다.")

for model_name, file_path in medqa_file_dict.items():
    if model_name not in medqa_index:
        print(f"모델 이름 '{model_name}'이 medqa_index에 없습니다.")
    else:
        try:
            with open(file_path, 'r') as f:
                pass  # 파일이 존재하는 경우는 무시
        except FileNotFoundError:
            print(f"'{file_path}' 파일이 존재하지 않습니다.")


with open('/home/lune/nas2/Projects/medical/medical_reasoning/sj/raw_data/medqa.json', 'r') as f:
    medqa_data = json.load(f)
with open('/home/lune/nas2/Projects/medical/medical_reasoning/sj/raw_data/ddxplus/chosen_rejected_consistency.json', 'r') as f:
    ddx_consistency = json.load(f)
with open('/home/lune/nas2/Projects/medical/medical_reasoning/sj/raw_data/ddxplus/chosen_rejected_explainability.json','r') as f:
    ddx_explainability = json.load(f)
with open('/home/lune/nas2/Projects/medical/medical_reasoning/sj/raw_data/ddxplus/chosen_rejected_grounding.json','r') as f:
    ddx_grounding = json.load(f)
with open('/home/lune/nas2/Projects/medical/medical_reasoning/sj/raw_data/ddxplus/chosen_rejected_strategic.json','r') as f:
    ddx_strategic = json.load(f)

def make_pair_for_medqa(input_dict, score_threshold=0.7):
    correct = input_dict['correct']
    incorrect = input_dict['incorrect']
    correct_pairs = []
    incorrect_pairs = []
    for i in correct['chosen']:
        for j in correct['reject']:
            if i[1] > j[1] and i[1] >= score_threshold and j[1] >= score_threshold:
                correct_pairs.append((i[0],j[0]))
    for i in incorrect['chosen']:
        for j in incorrect['reject']:
            if i[1] > j[1] and i[1] >= score_threshold and j[1] >= score_threshold:
                incorrect_pairs.append((i[0],j[0]))
    return correct_pairs, incorrect_pairs

def make_pair_for_ddx(input_dict, score_threshold=0.7):
    correct = input_dict['correct']
    incorrect = input_dict['incorrect']
    correct_pairs = []
    incorrect_pairs = []
    for i in correct['chosen']:
        for j in correct['rejected']:
            if float(i[1]) > float(j[1]) and float(i[1]) >= score_threshold and float(j[1]) >= score_threshold:
                correct_pairs.append((i[0],j[0]))
    for i in incorrect['chosen']:
        for j in incorrect['rejected']:
            if float(i[1]) > float(j[1]) and float(i[1]) >= score_threshold and float(j[1]) >= score_threshold:
                incorrect_pairs.append((i[0],j[0]))
    return correct_pairs, incorrect_pairs

def extract_input_output(data, index, is_medqa=True):
    generations = data['generations']
    get_data = generations[index]
    if is_medqa:
        if 'question' not in get_data['input']:
            return None, None, None, None
        if 'options' not in get_data['input']:
            return None, None, None, None
        if 'label' not in get_data:
            return None, None, None, None
        if 'output' not in get_data:
            return None, None, None, None
        if 'initial_response' not in get_data['output']:
            return None, None, None, None
        if 'initial_prediction' not in get_data['output']:
            return None, None, None, None
        if 'result' not in get_data:
            return None, None, None, None
        if get_data['output']['initial_response'] == None:
            return None, None, None, None
        if get_data['output']['initial_prediction'] == None:
            return None, None, None, None
        if get_data['result'][0] == None:
            return None, None, None, None
        get_input = get_data['input']['question'] + "\n" + get_data['input']['options']
        get_label = get_data['label']
        get_output = get_data['output']['initial_response'] + "\nInitial Prediction:\n" + get_data['output']['initial_prediction']
        get_result = get_data['result'][0]
        return get_input, get_label, get_output, get_result
    else:
        if 'evidences' not in get_data['input']:
            return None, None, None, None
        if 'options' not in get_data['input']:
            return None, None, None, None
        if 'label' not in get_data:
            return None, None, None, None
        if 'output' not in get_data:
            return None, None, None, None
        if 'initial_response' not in get_data['output']:
            return None, None, None, None
        if 'initial_prediction' not in get_data['output']:
            return None, None, None, None
        if 'result' not in get_data:
            return None, None, None, None
        if get_data['output']['initial_response'] == None:
            return None, None, None, None
        if get_data['output']['initial_prediction'] == None:
            return None, None, None, None
        if get_data['result'][0] == None:
            return None, None, None, None
        get_input = get_data['input']['evidences'] + "\n" + get_data['input']['options']
        get_label = get_data['label']
        get_output = get_data['output']['initial_response'] + "\nInitial Prediction:\n" + get_data['output']['initial_prediction']
        get_result = get_data['result'][0]
        return get_input, get_label, get_output, get_result
def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def convert_index_to_info(model_index, prob_index, is_medqa):
    if is_medqa:
        converted_index = find_key_by_value(medqa_index, model_index)
        json_file = medqa_file_dict[converted_index]
        with open(json_file, 'r') as f:
            data = json.load(f)
        get_input, get_label, get_output, get_result = extract_input_output(data, prob_index-1)
        return get_input, get_label, get_output, get_result
    else:
        converted_index = find_key_by_value(ddx_index, model_index)
        json_file = ddx_file_dict[converted_index]
        with open(json_file, 'r') as f:
            data = json.load(f)
        get_input, get_label, get_output, get_result = extract_input_output(data, prob_index-1, False)
        return get_input, get_label, get_output, get_result

def medqa_pair_generator(medqa_data, criteria):
    medqa_criteria = medqa_data[criteria]
    incorrect_list = []
    correct_list = []
    for i in medqa_criteria:
        problem_index = int(i)
        correct_pairs, incorrect_pairs = make_pair_for_medqa(medqa_criteria[i])
        for j in incorrect_pairs:
            k = j[0] #chosen model
            v = j[1] #reject model
            chosen_model = find_key_by_value(medqa_index, k)
            reject_model = find_key_by_value(medqa_index, v)
            pos_get_input, pos_get_label, pos_get_output, pos_get_result = convert_index_to_info(k, problem_index, True)
            neg_get_input, neg_get_label, neg_get_output, neg_get_result = convert_index_to_info(v, problem_index, True)
            if pos_get_input is not None and neg_get_input is not None:
                assert pos_get_label == neg_get_label
                assert pos_get_input == neg_get_input
                assert pos_get_result == neg_get_result
                incorrect_pair = {
                    'input': pos_get_input,
                    'label': pos_get_label,
                    'result': pos_get_result,
                    'chosen': pos_get_output,
                    'reject': neg_get_output,
                    'chosen_model': chosen_model,
                    'reject_model': reject_model
                }
                incorrect_list.append(incorrect_pair)
        for j in correct_pairs:
            k = j[0] #chosen model
            v = j[1] #reject model
            chosen_model = find_key_by_value(medqa_index, k)
            reject_model = find_key_by_value(medqa_index, v)
            pos_get_input, pos_get_label, pos_get_output, pos_get_result = convert_index_to_info(k, problem_index, True)
            neg_get_input, neg_get_label, neg_get_output, neg_get_result = convert_index_to_info(v, problem_index, True)
            if pos_get_input is not None and neg_get_input is not None:
                assert pos_get_label == neg_get_label
                assert pos_get_input == neg_get_input
                assert pos_get_result == neg_get_result
                correct_pair = {
                    'input': pos_get_input,
                    'label': pos_get_label,
                    'result': pos_get_result,
                    'chosen': pos_get_output,
                    'reject': neg_get_output,
                    'chosen_model': chosen_model,
                    'reject_model': reject_model
                }
                correct_list.append(correct_pair)
    return incorrect_list, correct_list

def ddx_pair_generator(ddx_data, criteria):
    if criteria == 'consistency':
        ddx_data = ddx_consistency
    elif criteria == 'explainability':
        ddx_data = ddx_explainability
    elif criteria == 'grounding':
        ddx_data = ddx_grounding
    elif criteria == 'strategic':
        ddx_data = ddx_strategic
    ddx_iter_data = ddx_data['questions']
    incorrect_list = []
    correct_list = []
    for i in ddx_iter_data:
        problem_index = int(i)
        correct_pairs, incorrect_pairs = make_pair_for_ddx(ddx_iter_data[i])
        for j in incorrect_pairs:
            k = j[0]
            v = j[1]
            chosen_model = find_key_by_value(ddx_index, k)
            reject_model = find_key_by_value(ddx_index, v)
            pos_get_input, pos_get_label, pos_get_output, pos_get_result = convert_index_to_info(k, problem_index, False)
            neg_get_input, neg_get_label, neg_get_output, neg_get_result = convert_index_to_info(v, problem_index, False)
            if pos_get_input is not None and neg_get_input is not None:
                assert pos_get_label == neg_get_label
                assert pos_get_input == neg_get_input
                assert pos_get_result == neg_get_result
                incorrect_pair = {
                    'input': pos_get_input,
                    'label': pos_get_label,
                    'result': pos_get_result,
                    'chosen': pos_get_output,
                    'reject': neg_get_output,
                    'chosen_model': chosen_model,
                    'reject_model': reject_model
                }
                incorrect_list.append(incorrect_pair)
        for j in correct_pairs:
            k = j[0]
            v = j[1]
            chosen_model = find_key_by_value(ddx_index, k)
            reject_model = find_key_by_value(ddx_index, v)
            pos_get_input, pos_get_label, pos_get_output, pos_get_result = convert_index_to_info(k, problem_index, False)
            neg_get_input, neg_get_label, neg_get_output, neg_get_result = convert_index_to_info(v, problem_index, False)
            if pos_get_input is not None and neg_get_input is not None:
                assert pos_get_label == neg_get_label
                assert pos_get_input == neg_get_input
                assert pos_get_result == neg_get_result
                correct_pair = {
                    'input': pos_get_input,
                    'label': pos_get_label,
                    'result': pos_get_result,
                    'chosen': pos_get_output,
                    'reject': neg_get_output,
                    'chosen_model': chosen_model,
                    'reject_model': reject_model
                }
                correct_list.append(correct_pair)
    return incorrect_list, correct_list


ddx_type = ['consistency', 'explainability', 'grounding', 'strategic']
for types in ddx_type:
    incorrect_pairs, correct_pairs = ddx_pair_generator(ddx_data, types)
    incorrect_dict = {'data': incorrect_pairs}
    correct_dict = {'data': correct_pairs}
    with open(f'/home/lune/nas2/Projects/medical/medical_reasoning/sj/generated_data/ddxplus/{types}_incorrect.json', 'w') as f:
        json.dump(incorrect_dict, f, indent=4, ensure_ascii=False)
    with open(f'/home/lune/nas2/Projects/medical/medical_reasoning/sj/generated_data/ddxplus/{types}_correct.json', 'w') as f:
        json.dump(correct_dict, f, indent=4, ensure_ascii=False)

med_type = ['consistency', 'explainability', 'correctness']
for types in med_type:
    incorrect_pairs, correct_pairs = medqa_pair_generator(medqa_data, types)
    incorrect_dict = {'data': incorrect_pairs}
    correct_dict = {'data': correct_pairs}
    print(len(incorrect_pairs))
    print(len(correct_pairs))
    with open(f'/home/lune/nas2/Projects/medical/medical_reasoning/sj/generated_data/medqa/{types}_incorrect.json', 'w') as f:
        json.dump(incorrect_dict, f, indent=4, ensure_ascii=False)
    with open(f'/home/lune/nas2/Projects/medical/medical_reasoning/sj/generated_data/medqa/{types}_correct.json', 'w') as f:
        json.dump(correct_dict, f, indent=4, ensure_ascii=False)
        
total_list = []
from copy import deepcopy
med_type = ['consistency', 'explainability', 'correctness']
ddx_type = ['consistency', 'explainability', 'grounding', 'strategic']
ddx_dir = '/home/lune/nas2/Projects/medical/medical_reasoning/sj/generated_data/ddxplus'
med_dir = '/home/lune/nas2/Projects/medical/medical_reasoning/sj/generated_data/medqa'
for med in med_type:
    with open(f'{med_dir}/{med}_incorrect.json', 'r') as f:
        incorrect_data = json.load(f)
    with open(f'{med_dir}/{med}_correct.json', 'r') as f:
        correct_data = json.load(f)

    for d in incorrect_data['data']:
        copy_data = deepcopy(d)
        new_data = {
            **copy_data,
            'data_type': 'medqa',
            'criteria': med,
            'correctness': False
        }
        total_list.append(new_data)
    for d in correct_data['data']:
        copy_data = deepcopy(d)
        new_data = {
            **copy_data,
            'data_type': 'medqa',
            'criteria': med,
            'correctness': True
        }
        total_list.append(new_data)
for ddx in ddx_type:
    with open(f'{ddx_dir}/{ddx}_incorrect.json', 'r') as f:
        incorrect_data = json.load(f)
    with open(f'{ddx_dir}/{ddx}_correct.json', 'r') as f:
        correct_data = json.load(f)
    for d in incorrect_data['data']:
        copy_data = deepcopy(d)
        new_data = {
            **copy_data,
            'data_type': 'ddx',
            'criteria': ddx,
            'correctness': False
        }
        total_list.append(new_data)
    for d in correct_data['data']:
        copy_data = deepcopy(d)
        new_data = {
            **copy_data,
            'data_type': 'ddx',
            'criteria': ddx,
            'correctness': True
        }
        total_list.append(new_data)
train_dataset = load_dataset("json", data_files=f'/home/lune/nas2/Projects/medical/medical_reasoning/sj/generated_data/total_data.json', field="data")
dataset = DatasetDict({'train': train_dataset['train']})

dataset.push_to_hub("DLI-Lab/Medical_reward_bench")