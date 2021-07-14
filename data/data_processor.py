'''
清洗心聆数据，将其用于poly-encoder架构

author: Qiu Huachuan
date: 2021-6-22
'''

import ujson

with open('./data_v3.json', 'r', encoding='utf-8') as f:
    data = ujson.load(f)

file_names = ['train', 'valid', 'test']

for name in file_names:
    if name == 'train':
        output = []
        for index, item in enumerate(data[name]):
            context = ''

            contexts = item['history']
            for element in contexts:
                new_element = element.replace('咨询师：', '[therapist]').replace('来访者：', '[client]')
                context += new_element

            responses = item['responses']
            for idx, elements in enumerate(responses):
                response = ''
                if item['label'] == idx:
                    for sub_element in elements:
                        new_sub_element = sub_element.replace('咨询师：', '').replace('来访者：', '')
                        response += new_sub_element
                    new_item = {
                        'context': context,
                        'candidate': '[therapist]' + response,
                        'label': 1
                    }
                    output.append(new_item)
        with open(f'{name}.json', 'w', encoding='utf-8') as f_0:
            ujson.dump(output[:32], f_0, ensure_ascii=False, indent=4)
        print(len(output))
    else:
        output = []
        for index, item in enumerate(data[name]):
            context = ''

            contexts = item['history']
            for element in contexts:
                new_element = element.replace('咨询师：', '[therapist]').replace('来访者：', '[client]')
                context += new_element
        
            responses = item['responses']
            for idx, elements in enumerate(responses):
                response = ''
                for sub_element in elements:
                    new_sub_element = sub_element.replace('咨询师：', '').replace('来访者：', '[client]')
                    response += new_sub_element

                new_item = {
                    'context': context,
                    'candidate': '[therapist]' + response,
                    'label': 1 if item['label'] == idx else 0
                }
                output.append(new_item)
        with open(f'{name}.json', 'w', encoding='utf-8') as f_0:
            ujson.dump(output, f_0, ensure_ascii=False)
