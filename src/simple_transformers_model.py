import torch.nn as nn


class SimpleTransformersModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        
        self.base_model = base_model
        
        self.belong_out = nn.Sequential(
            nn.Linear(base_model.belong_linear[-2].out_features, 1),
            nn.Sigmoid()
        )
        self.burden_out = nn.Sequential(
            nn.Linear(base_model.burden_linear[-2].out_features, 1),
            nn.Sigmoid()
        )      
        
        
    def forward(self, x):
        base_out = self.base_model(x)
        
        belong_out = self.belong_out(base_out[0])
        burden_out = self.burden_out(base_out[1])
                
        return belong_out.squeeze(), burden_out.squeeze()
