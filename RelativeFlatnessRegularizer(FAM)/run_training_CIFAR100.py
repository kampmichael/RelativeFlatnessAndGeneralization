import time
import torch
import torch.nn.functional as F
import torchmetrics.classification.accuracy as torch_acc
from torch import nn

import torchvision.models as t_models
#from models import efficientnet_b7_fam, efficientnet_b0_fam
from FAMloss import FAMreg, LayerHessian
from data import CIFAR100DataModule

from torchvision.models.efficientnet import efficientnet_b7

# PyTorch Lightning
import pytorch_lightning as pl
pl.seed_everything(hash("setting random seeds") % 2 ** 32 - 1)



LR = 0.00001
BS = 64
EPOCHS = 100
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9
NESTEROV = True

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

class EffNet_B7(pl.LightningModule):
    def __init__(self):
        super().__init__()

        #model = efficientnet_b0_fam(weights=t_models.EfficientNet_B0_Weights)
        model = efficientnet_b7(weights=t_models.EfficientNet_B7_Weights)
        #set_parameter_requires_grad(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 100)
        #model.eval()
        self.core = model
        
        self.hparams["network"] = 'EffNet_B7' # 'EffNet_B0'
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torch_acc.Accuracy()
        self.valid_acc = torch_acc.Accuracy()
        self.test_acc = torch_acc.Accuracy()

    def loss(self, xs, ys):
        outputs = self.core(xs)
        ys = ys.type(torch.LongTensor).cuda()
        criterion = F.cross_entropy
        
        loss = criterion(outputs, ys)

        self.logger.experiment.config.loss = 'cross entropy'

        return outputs, loss

    # lightning hook to add an optimizer
    def configure_optimizers(self):
        lr = LR
        #self.logger.experiment.config.optimizer = 'SGD'
        self.logger.experiment.config.optimizer = 'Adam'
        self.logger.experiment.config.lr = lr
        #self.logger.experiment.config.momentum = MOMENTUM
        self.logger.experiment.config.weight_decay = WEIGHT_DECAY
        #self.logger.experiment.config.scheduler = 'one_cycle_lr'
        #self.logger.experiment.config.nesterov = NESTEROV
        optimizer = torch.optim.Adam(
            self.core.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        #optimizer = torch.optim.SGD(
        #    self.core.parameters(),
        #    lr=lr,
        #    momentum=MOMENTUM,
        #    weight_decay=WEIGHT_DECAY,
        #    nesterov = NESTEROV
        #)
        #scheduler_dict = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        return {"optimizer": optimizer} #, "lr_scheduler": scheduler_dict}

    # lightning hook to make a training step
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        outputs, loss = self.loss(xs, ys)
        preds = torch.argmax(outputs, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)

        return loss

    # lightning hook to make a testing run
    def test_step(self, batch, batch_idx):
        xs, ys = batch
        outputs, loss = self.loss(xs, ys)
        preds = torch.argmax(outputs, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    # lightning hook to make a validation run
    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        outputs, loss = self.loss(xs, ys)
        preds = torch.argmax(outputs, 1)

        self.valid_acc(preds, ys)
        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)

        return outputs


if __name__ == "__main__":
    # setup data
    cifar100 = CIFAR100DataModule(data_dir="datasets", batch_size=BS)
    cifar100.prepare_data()
    cifar100.setup()

    # setup model
    model = EffNet_B7()

    trainer = pl.Trainer(
        precision=16,
        check_val_every_n_epoch=5,
        log_every_n_steps=50,  # set the logging frequency
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,  # number of epochs
        deterministic=True,  # keep it deterministic
    )

    # fit the model
    trainer.fit(model, cifar100)

    # evaluate the model on a test set
    trainer.test(model, datamodule=cifar100, ckpt_path=None)  # uses last-saved model

    torch.save(model.core.state_dict(), 'trained_models/'+model.core.saveName()+"_"+str(int(time.time()))+".model")

