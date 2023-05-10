import torch.nn as nn
from transformers import AutoModel, AutoConfig
from collections import OrderedDict


class BaseTransformersModel(nn.Module):
    def __init__(self, lm_name, shared_hidden_size=[48], belong_hidden_size=[16], burden_hidden_size=[16]):
        super().__init__()
        
        self.lm_name = lm_name
        self.lm = AutoModel.from_pretrained(lm_name, use_auth_token=True)
        self.lm_config = AutoConfig.from_pretrained(lm_name, use_auth_token=True)
        self.lm.requires_grad_(False)
        
        if len(shared_hidden_size):
            self.shared_linear = nn.Sequential(
                nn.Linear(self.lm_config.hidden_size,shared_hidden_size[0]),
                nn.ReLU(),
            )
            for i in range(1, len(shared_hidden_size) - 1):
                self.shared_linear.append(nn.Linear(shared_hidden_size[i], shared_hidden_size[i+1]))
                self.shared_linear.append(nn.ReLU())
        else:
            self.shared_linear = None
            shared_hidden_size = [self.lm_config.hidden_size]
                
        self.belong_linear = nn.Sequential(
            nn.Linear(shared_hidden_size[-1],belong_hidden_size[0]),
            nn.ReLU(),
        )
        for i in range(1, len(belong_hidden_size) - 1):
            self.belong_linear.append(nn.Linear(belong_hidden_size[i], belong_hidden_size[i+1]))
            self.belong_linear.append(nn.ReLU())
        
        # self.belong_linear.append(nn.Linear(belong_hidden_size[-1], 1))
        # self.belong_linear.append(nn.Sigmoid())
        
        self.burden_linear = nn.Sequential(
            nn.Linear(shared_hidden_size[-1],burden_hidden_size[0]),
            nn.ReLU(),
        )
        for i in range(1, len(burden_hidden_size) - 1):
            self.burden_linear.append(nn.Linear(burden_hidden_size[i], burden_hidden_size[i+1]))
            self.burden_linear.append(nn.ReLU())
        
        
        # self.burden_linear.append(nn.Linear(burden_hidden_size[-1], 1))
        # self.burden_linear.append(nn.Sigmoid())
        
        
        
    def forward(self, x):
        lm_out = self.lm(
            input_ids=x["input_ids"], attention_mask=x["attention_mask"]
        ).pooler_output
       
        if self.shared_linear:
            shared_linear_out = self.shared_linear(lm_out)
        else:
            shared_linear_out = lm_out
            
        belong_out = self.belong_linear(shared_linear_out)
        burden_out = self.burden_linear(shared_linear_out)
                
        return belong_out, burden_out
