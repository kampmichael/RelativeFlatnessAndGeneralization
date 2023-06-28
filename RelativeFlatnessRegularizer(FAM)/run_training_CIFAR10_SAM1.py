import time
import torch
import torch.nn.functional as F
import torchmetrics.classification.accuracy as torch_acc

from SAMloss1 import SAM, enable_running_stats, disable_running_stats
from models import ResNet18
from data import CIFAR10DataModule

# PyTorch Lightning
import pytorch_lightning as pl
pl.seed_everything(hash("setting random seeds") % 2 ** 32 - 1)


LR = 0.05
BS = 64
EPOCHS = 200

RHO = 0.05

class LitRes18Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # 3 channels in CIFAR10 images
        self.core = ResNet18(num_classes=10)

        self.hparams["network"] = 'ResNet18'
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
        optimizer = SAM(self.core.parameters(), base_optimizer, rho=rho, lr=lr, momentum=0.9)
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
    cifar10 = CIFAR10DataModule(data_dir="datasets", batch_size=BS)
    cifar10.prepare_data()
    cifar10.setup()

    # setup model
    model = LitRes18Net()

    trainer = pl.Trainer(
        log_every_n_steps=101,  # set the logging frequency
        gpus=1,  # use all GPUs
        max_epochs=EPOCHS,  # number of epochs
        deterministic=True,  # keep it deterministic
    )

    # fit the model
    trainer.fit(model, cifar10)

    # evaluate the model on a test set
    trainer.test(model, datamodule=cifar10, ckpt_path=None)  # uses last-saved model

    torch.save(model.core.state_dict(), 'trained_models/'+model.core.saveName()+"_"+str(int(time.time()))+".model")


