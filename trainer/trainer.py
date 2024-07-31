import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from utils.train_utils import get_dataloader
from transformers import AutoTokenizer



class Trainer(BaseTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)

        # dataloaders
        self.dataloader = get_dataloader(config) if config.mode == 'train' else None # {'train': dataloader, 'valid': dataloader}

        # main process
        self.rank_zero = True if not self.ddp or (self.ddp and device == 0) else False

        # initialize trainer
        self._init_trainer()

        # criterion
        self.cross_entropy = nn.CrossEntropyLoss()


    def _training_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """        
        
        output = self.model(
            input_ids=model_inputs['input_ids'],
            token_type_ids=model_inputs['token_type_ids'],
            attention_mask=model_inputs['attention_mask'],
        )
        
        logits = output.logits # batch, num labels
        loss = self.cross_entropy(logits.view(-1, self.config.num_labels), model_inputs['labels'].view(-1))

        self._backward_step(loss)

        return loss.item()


    @torch.no_grad()
    def _validation_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """
        output = self.model(
            input_ids=model_inputs['input_ids'],
            token_type_ids=model_inputs['token_type_ids'],
            attention_mask=model_inputs['attention_mask']
        )
        
        logits = output.logits           # batch, num labels
        loss = self.cross_entropy(logits.view(-1, self.config.num_labels), model_inputs['labels'].view(-1))

        return loss.item()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, cache_dir=self.config.cache_dir)
        while True:
            text = input("sentence (if you want to end the test, Enter exit.) >")
            if not text or text in ["exit", "exit()"]: break

            inputs = tokenizer(
                text,
                return_tensors='pt',
                add_special_tokens=True,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
            )
            inputs = {k:v.to(self.device) for k, v in inputs.items()}
            output = self.model(**inputs)
            
            softmax_output = F.softmax(output.logits, dim=1)
            print(softmax_output)
            idx = torch.argmax(softmax_output).item()
            if idx == 1:
                print("This sentence is a positive.\n")
            else:
                print("This sentence is a negative.\n")