'''
This file is used for loading data

Author: Huachuan Qiu
'''

import ujson

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

tokenizer = BertTokenizer.from_pretrained(
    './models', never_split=['[therapist]', '[client]'])


def dataloader(file_name):
    with open(f'./data/{file_name}.json', 'r', encoding='utf-8') as f:
        data = ujson.load(f)
    labels = []
    context_input_ids = []
    context_attention_mask = []
    candidate_input_ids = []
    candidate_attention_mask = []
    for item in data:
        context_encoded_dict = tokenizer.encode_plus(
            item['context'],
            add_special_tokens=True,
            max_length=500,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
        context_input_ids.append(context_encoded_dict['input_ids'])
        context_attention_mask.append(context_encoded_dict['attention_mask'])

        candidate_encoded_dict = tokenizer.encode_plus(
            item['candidate'],
            add_special_tokens=True,
            max_length=500,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
        candidate_input_ids.append(candidate_encoded_dict['input_ids'])
        candidate_attention_mask.append(candidate_encoded_dict['attention_mask'])

        labels.append(item['label'])
    context_input_ids = torch.cat(context_input_ids, dim=0)
    context_attention_mask = torch.cat(context_attention_mask, dim=0)
    candidate_input_ids = torch.cat(candidate_input_ids, dim=0)
    candidate_attention_mask = torch.cat(candidate_attention_mask, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(context_input_ids, context_attention_mask,
                            candidate_input_ids, candidate_attention_mask, labels)

    if file_name == 'train':
        dataset_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    elif file_name == 'valid':
        dataset_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    else:
        dataset_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    return dataset_loader
