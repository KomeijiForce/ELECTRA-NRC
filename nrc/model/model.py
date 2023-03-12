import numpy as np

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForPreTraining, AutoModelForMaskedLM, GPT2LMHeadModel

from copy import deepcopy


class InferenceModel(nn.Module):
    
    def __init__(self, model_path, **kwargs):
        
        super().__init__()
    
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        if any(["MaskedLM" in architecture for architecture in self.config.architectures]):
            self.action = "mlm"
        elif any(["LMHeadModel" in architecture for architecture in self.config.architectures]):
            self.action = "lm"
        elif any(["ForPreTraining" in architecture for architecture in self.config.architectures]):
            self.action = "rtd"
        
        if self.action == 'mlm':
            ModelType = AutoModelForMaskedLM
        elif self.action == 'lm':
            ModelType = GPT2LMHeadModel
        elif self.action == 'rtd':
            ModelType = AutoModelForPreTraining
        self.model = ModelType.from_pretrained(model_path).to(self.device)
        
    def inference(self, question, answer, target):
        
        with torch.no_grad():
            if self.action == 'rtd':
                return self.rtd(question, answer, target)
            elif self.action == 'lm':
                return self.lm(question, answer, target)
            elif self.action == 'mlm':
                return self.mlm(question, answer, target)


    def rtd(self, question, answer, target):

        len_q = len(self.tok.tokenize(question))

        items = self.tok([question], [answer])

        for key in items.keys():
            items[key] = torch.LongTensor(items[key]).to(self.device)

        logit = -self.model(**items).logits[0, 1:-1]
        if target == 'q':
            logit = logit[:len_q]
        elif target == 'a':
            logit = logit[len_q+1:]

        return logit.mean().item()


    def lm(self, question, answer, target):

        len_q = len(self.tok.tokenize(question))
    
        items = self.tok(question + " " + answer)

        for key in items.keys():
            items[key] = torch.LongTensor(items[key]).to(self.device)

        logit = self.model(**items).logits
        logit = logit[:-1].softmax(-1).gather(1, items['input_ids'][1:].unsqueeze(-1)).log()
        if target == 'q':
            logit = logit[:len_q-1]
        elif target == 'a':
            logit = logit[len_q-1:]

        return logit.mean().item()


    def mlm(self, question, answer, target):

        len_q = len(self.tok.tokenize(question))
        
        ids_masked = []
        
        tokens = self.tok.tokenize(question + self.tok.sep_token + answer)
        token_ids = self.tok.convert_tokens_to_ids(tokens)
        
        for idx, token in enumerate(token_ids):
            token_ids_ = deepcopy(token_ids)
            token_ids_[idx] = self.tok.mask_token_id
            token_ids_ = [self.tok.cls_token_id] + token_ids_ + [self.tok.sep_token_id]
            ids_masked.append(token_ids_)

        ids_masked = torch.LongTensor(ids_masked).to(self.device)

        logit = self.model(ids_masked).logits
        logit = [logit[idx, idx+1].softmax(-1)[token_idx].log().item() for idx, token_idx in enumerate(token_ids)]
        if target == 'q':
            logit = logit[:len_q]
        elif target == 'a':
            logit = logit[len_q+1:]

        return np.mean(logit)