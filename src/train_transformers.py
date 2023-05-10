import argparse
import uuid
import wandb
import pytorch_lightning as pl
import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger

from transformers_model import BaseTransformersModel
from simple_transformers_model import SimpleTransformersModel
from stacked_transformers_model import StackedTransformersModel
from transformers_trainer import TransformersTrainer


pl.seed_everything(42)



def objective(config, experiment):
    name = str(uuid.uuid1())
    wandb.config['name'] = name
    acc_ckpt = pl.callbacks.ModelCheckpoint(
        monitor="avg_accuracy",
        mode="max",
        verbose=True,
        dirpath="../checkpoints/",
        filename=f"{name}",
    )
    
    bt_models = []
    for model_name in config.model_names:
        btmodel = BaseTransformersModel(model_name, shared_hidden_size=config.shared_hidden, belong_hidden_size=config.b_hidden, burden_hidden_size=config.b_hidden)
        bt_models.append(btmodel)
    
    if len(bt_models) == 1:
        stmodel = SimpleTransformersModel(bt_models[0])
    else:
        stmodel = StackedTransformersModel(bt_models)
    
    model = TransformersTrainer(stmodel, text_col="text", batch_size=64, preprocess=config.preprocess)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        precision=16,
        max_epochs=20,
        auto_select_gpus=True,
        # strategy=plugin,
        callbacks=[acc_ckpt],
        fast_dev_run=False,
        detect_anomaly=False,
        logger=WandbLogger(experiment=experiment, name=name),
    )

    trainer.fit(model)
    test_results = trainer.test(model, ckpt_path=f'../checkpoints/{name}.ckpt')[0]
    preds = trainer.predict(model, dataloaders=trainer.test_dataloaders[0], ckpt_path=f'../checkpoints/{name}.ckpt')
    np.save(f'../predictions/{name}.npy', torch.concat(preds, dim=1).detach().cpu().numpy())
    return test_results
    
def main(args):
    experiment = wandb.init(project='mental2', config=args)
    score = objective(wandb.config, experiment)
    wandb.log(score)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_names', type=list, default=['j-hartmann/emotion-english-distilroberta-base', 'mlaricheva/roberta-psych', 'AIMH/mental-roberta-large'])
    parser.add_argument('--shared_hidden', type=list, default=[48])
    parser.add_argument('--b_hidden', type=list, default=[32])
    parser.add_argument('--preprocess', type=bool, default=False)
    main(parser.parse_args())
    

# sweep_id = wandb.sweep(sweep=sweep_configuration, project='mental2')
# wandb.agent(sweep_id, function=main)