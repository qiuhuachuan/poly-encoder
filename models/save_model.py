'''
save bert-base-chinese model to local directory of './models'

Author: Huachuan Qiu
'''

import os

import torch
from transformers import BertModel, BertTokenizer, CONFIG_NAME, WEIGHTS_NAME

output_dir = './'

model = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# modify two special tokens in the vocab
tokenizer.vocab['[therapist]'] = tokenizer.vocab.pop('[unused1]')
tokenizer.vocab['[client]'] = tokenizer.vocab.pop('[unused2]')
# save vocab.txt
tokenizer.save_vocabulary(output_dir)

# save pytorch_model.bin
torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

# save config.json file
getattr(model, 'module', model).config.to_json_file(
    os.path.join(output_dir, CONFIG_NAME)
)
