import time
import torch
import torch.nn.functional as F
import torchmetrics.classification.accuracy as torch_acc

from SAMloss1 import SAM, enable_running_stats, disable_running_stats
import torchvision.models as t_models
from models import efficientnet_b7_fam, efficientnet_b0_fam
from FAMloss import FAMreg, LayerHessian
from data import CIFAR100DataModule

# PyTorch Lightning
import pytorch_lightning as pl
pl.seed_everything(hash("setting random seeds") % 2 ** 32 - 1)


LR = 1e-5
BS = 64
EPOCHS = 100
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
NESTEROV = True

RHO = 0.05

class EffNet_B7(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        model = efficientnet_b7_fam(weights=t_models.EfficientNet_B7_Weights)
        self.core = model

        self.hparams["network"] = 'EffNet_B7'
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torch_acc.Accuracy()
        self.valid_acc = torch_acc.Accuracy()
        self.test_acc = torch_acc.Accuracy()

    def loss(self, xs, ys):
        outputs = self.core(xs)
        ys = ys.type(torch.LongTensor).cuda()
        # have to use functional loss in here!
        loss = F.cross_entropy(outputs, ys)
        self.logger.experiment.config.loss = 'CrossEntropy'
        return outputs, loss

    # lightning hook to add an optimizer
    def configure_optimizers(self):
        lr = LR
        rho = RHO
        self.logger.experiment.config.optimizer = 'SAM1 SGD'
        self.logger.experiment.config.lr = lr
        self.logger.experiment.config.rho = rho
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(self.core.parameters(), base_optimizer, rho=rho, lr=lr, momentum=MOMENTUM, nesterov=NESTEROV,
                        weight_decay=WEIGHT_DECAY)
        return {"optimizer": optimizer}

    # lightning hook to make a training step
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        xs, ys = batch

        # first forward-backward pass
        self.core.apply(enable_running_stats)
        outputs, loss = self.loss(xs, ys)
        self.manual_backward(loss)
        opt.first_step(zero_grad=True)

        preds = torch.argmax(outputs, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)

        # second forward-backward pass
        self.core.apply(disable_running_stats)
        _, loss1 = self.loss(xs, ys)
        self.manual_backward(loss1)  # make sure to do a full forward pass
        opt.second_step(zero_grad=True)
        
        # make scheduler step
        #sch = self.lr_schedulers()
        #if self.trainer.is_last_batch:
        #    sch.step()
            
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

    #torch.save(model.core.state_dict(), 'trained_models/'+model.core.saveName()+"_"+str(int(time.time()))+".model")


