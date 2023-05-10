import torch
import torch.nn as nn


class StackedTransformersModel(nn.Module):
    def __init__(self, base_models):
        super().__init__()
        
        self.base_models = base_models
        
        for i, base_model in enumerate(self.base_models):
            setattr(self, f"base_model_{i}", base_model)
        
        belong_dim = 0
        burden_dim = 0
        for i, base_model in enumerate(self.base_models):
            belong_dim += getattr(self, f"base_model_{i}").belong_linear[-2].out_features
            burden_dim += getattr(self, f"base_model_{i}").burden_linear[-2].out_features
        
        self.belong_out = nn.Sequential(
            nn.Linear(belong_dim, belong_dim//2),
            nn.Linear(belong_dim//2, 1),
            nn.Sigmoid()
        )
        self.burden_out = nn.Sequential(
            nn.Linear(burden_dim, burden_dim//2),
            nn.Linear(burden_dim//2, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        base_belong_outs = []
        base_burden_outs = []
        for i, base_model in enumerate(self.base_models):
            base_out = getattr(self, f"base_model_{i}")(x)
            base_belong_outs.append(base_out[0])
            base_burden_outs.append(base_out[1])
        
        belong_concat = torch.cat(base_belong_outs, dim=1)
        burden_concat = torch.cat(base_burden_outs, dim=1)
        
        
        belong_out = self.belong_out(belong_concat)
        burden_out = self.burden_out(burden_concat)
        
        return belong_out.squeeze(), burden_out.squeeze()
