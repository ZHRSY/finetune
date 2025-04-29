import json
from datasets import Dataset





class LocalJsonDataset:
    def __init__(self, json_file, data_template, tokenizer, max_seq_length=2048):
        self.json_file = json_file
        self.data_template = data_template
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = self.load_dataset()

    def load_dataset(self):
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        for item in data:
            text = self.data_template.format(item['input'], item['output']) + self.tokenizer.eos_token
            texts.append(text)

        dataset_dict = {
            'text': texts  # 添加'text'字段以适配SFTTrainer
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    def get_dataset(self):
        return self.dataset