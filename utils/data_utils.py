import torch
import pandas as pd

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.data = pd.read_csv(config.data_path)
        self.tokenizer = tokenizer
        self.max_length = config.max_length

    def __getitem__(self, index):
        row = self.data.iloc[index]
        document, label = row['document'], row['label']

        document = self.tokenizer(
            document,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            "input_ids": document.input_ids.squeeze(),
            "token_type_ids": document.token_type_ids.squeeze(),
            "labels": torch.LongTensor([label]) 
        }


    def __len__(self):
        return len(self.data)