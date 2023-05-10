import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers_dataset import TransformersDataset

class TransformersTrainer(pl.LightningModule):
    def __init__(self, transformers_model, batch_size=8, optim='adam', lr=1e-3, text_col='text', preprocess=False):
        super().__init__()
        
        self.batch_size = batch_size
        self.optim = optim
        self.lr = lr
        self.preprocess = preprocess
        
        self.model = transformers_model
        
        # self.loss = nn.BinaryCrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')
        self.f1 = F1Score(task='binary')
        self.precision_s = Precision(task='binary')
        self.recall = Recall(task='binary')
        
        self.text_col = text_col
        self.lm_name = 'roberta-base'
        self.criterion = nn.CrossEntropyLoss()


    def training_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_op, burden_op = self.model.forward(x)
        
        belong_loss = self.criterion(belong_op, belong_labels)
        burden_loss = self.criterion(burden_op, burden_labels)
        loss = belong_loss + burden_loss
    
        self.log("training_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_op, burden_op = self.model.forward(x)
        
        self.log_dict(
            self.calc_metrics(belong_op, burden_op, belong_labels, burden_labels),
            on_epoch=True,
            prog_bar=True,
        )
            
    
    def predict_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_logits, burden_logits = self.model.forward(x)
        # return {
        #     "belong": belong_logits, #torch.round(belong_logits),
        #     "burden": burden_logits, #torch.round(burden_logits),
        # }
        return torch.vstack((belong_logits, burden_logits))
    
    
    def test_step(self, batch, batch_idx):
        x, belong_labels, burden_labels = batch
        belong_logits, burden_logits = self.model.forward(x)
        
        self.log_dict(self.calc_metrics(belong_logits, burden_logits, belong_labels, burden_labels))
        
    
    def calc_metrics(self, belong_logits, burden_logits, belong_labels, burden_labels):
        belong_accuracy = self.accuracy(belong_logits, belong_labels.int())
        burden_accuracy = self.accuracy(burden_logits, burden_labels.int())
        avg_accuracy = (burden_accuracy+belong_accuracy)/2
        
        belong_precision = self.precision_s(belong_logits, belong_labels.int())
        burden_precision = self.precision_s(burden_logits, burden_labels.int())
        avg_precision = (belong_precision+burden_precision)/2
        
        belong_recall = self.recall(belong_logits, belong_labels.int())
        burden_recall = self.recall(burden_logits, burden_labels.int())
        avg_recall = (burden_recall+belong_recall)/2
        
        belong_f1 = self.f1(belong_logits, belong_labels.int())
        burden_f1 = self.f1(burden_logits, burden_labels.int())
        avg_f1 = (burden_f1+belong_f1)/2
        
        return {
            'belong_accuracy': belong_accuracy,
            'burden_accuracy': burden_accuracy,
            'avg_accuracy': avg_accuracy,
            'belong_precision': belong_precision,
            'burden_precision': burden_precision,
            'avg_precision': avg_precision,
            'belong_recall': belong_recall,
            'burden_recall': burden_recall,
            'avg_recall': avg_recall,
            'belong_f1': belong_f1,
            'burden_f1': burden_f1,
            'avg_f1': avg_f1,
        }


    def configure_optimizers(self):
        if self.optim == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == 'adamax':
            return torch.optim.Adamax(self.parameters(), lr=self.lr)
        elif self.optim == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

    def train_dataloader(self):
        ds = TransformersDataset(
            path= "../data/train_data_pre.csv" if self.preprocess else "../data/train_data.csv",
            text_col=self.text_col,
            lm_name=self.lm_name,
        )
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    
    def val_dataloader(self):
        ds = TransformersDataset(
            path= "../data/val_data_pre.csv" if self.preprocess else "../data/val_data.csv",
            text_col=self.text_col,
            lm_name=self.lm_name,
        )
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    
    def test_dataloader(self):
        ds = TransformersDataset(
            path= "../data/test_data_pre.csv" if self.preprocess else "../data/test_data.csv",
            text_col=self.text_col,
            lm_name=self.lm_name,
        )
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)