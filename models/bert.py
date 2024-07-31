import torch.nn as nn

from transformers import BertForSequenceClassification, AutoConfig


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(config.model_path, num_labels=config.num_labels, cache_dir=config.cache_dir)
        self.config = self.model.config


    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return output
    


