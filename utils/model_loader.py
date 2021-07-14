from transformers import BertModel, BertConfig

model_path = './models'
bert = BertModel.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)