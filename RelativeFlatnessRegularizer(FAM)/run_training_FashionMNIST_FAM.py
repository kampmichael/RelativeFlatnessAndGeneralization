import time
import torch
import torch.nn.functional as F
import torchmetrics.classification.accuracy as torch_acc

from models import WideResNet_28_10
from data import FashionMNISTDataModule
from FAMloss import FAMreg, LayerHessian

# PyTorch Lightning
import pytorch_lightning as pl
pl.seed_everything(hash("setting random seeds") % 2 ** 32 - 1)


LR = 0.05
BS = 64
EPOCHS = 200

feature_layer_id = 106 + 2 # penultimate layer
lmb = 0.1

class LitWRNNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.core = WideResNet_28_10(num_classes=10)

        self.hparams["network"] = 'WideResNet28_10 with bottleneck'
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torch_acc.Accuracy()
        self.valid_acc = torch_acc.Accuracy()
        self.test_acc = torch_acc.Accuracy()

    def loss(self, xs, ys):
        outputs = self.core(xs)
        ys = ys.type(torch.LongTensor).cuda()
        criterion = F.cross_entropy

        if self.training: # compute FAM regularized loss
            layer_hessian = LayerHessian(self.core, feature_layer_id, criterion, method="autograd_fast")
            loss = criterion(outputs, ys) + lmb * FAMreg(xs, ys, layer_hessian, norm_function='neuronwise')
        else:
            loss = criterion(outputs, ys)

        self.logger.experiment.config.regulariz_lambda = lmb
        self.logger.experiment.config.loss = 'FAM, neuronwise autograd_fast'

        return outputs, loss

    # lightning hook to add an optimizer
    def configure_optimizers(self):
        lr = LR
        self.logger.experiment.config.optimizer = 'SGD'
        self.logger.experiment.config.lr = lr
        self.logger.experiment.config.scheduler = 'multistep_lr'
        optimizer = torch.optim.SGD(self.core.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(elem * EPOCHS) for
                                                                            elem in [0.4, 0.6, 0.8]], gamma=0.2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
    fmnist = FashionMNISTDataModule(data_dir="datasets", batch_size=BS)
    fmnist.prepare_data()
    fmnist.setup()

    # setup model
    model = LitWRNNet()

    trainer = pl.Trainer(
        log_every_n_steps=101,  # set the logging frequency
        gpus=1,  # use all GPUs
        max_epochs=EPOCHS,  # number of epochs
        deterministic=True,  # keep it deterministic
    )

    # fit the model
    trainer.fit(model, fmnist)

    # evaluate the model on a test set
    trainer.test(model, datamodule=fmnist, ckpt_path=None)  # uses last-saved model

    #torch.save(model.core.state_dict(), 'trained_models/'+model.core.saveName()+"_"+str(int(time.time()))+".model")

