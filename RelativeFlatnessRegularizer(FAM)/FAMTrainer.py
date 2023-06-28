from typing import Union, Optional, List, Dict, Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from torch.utils.data import Dataset

from FAMloss import LayerHessian, FAMreg


class FAMTrainer(Trainer):

    def __init__(self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        feature_layer_id = None,
        fam_lambda = None
        
    ):
        super().__init__(model, 
                        args, 
                        data_collator, 
                        train_dataset, 
                        eval_dataset,
                        tokenizer,
                        model_init,
                        compute_metrics,
                        callbacks,
                        optimizers,
                        preprocess_logits_for_metrics
                        )
        self.feature_layer_id = feature_layer_id
        self.fam_lambda = fam_lambda


    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        criterion = F.cross_entropy
        
        if self.model.training: # compute FAM regularized loss!
            # if padding, stride and input shape are given, the layer is treated as Conv2D layer
            layer_hessian = LayerHessian(model, self.feature_layer_id, criterion, method="functorch")#, padding=1, stride=1, input_shape=(32,32))
            reg = FAMreg(inputs, labels, layer_hessian)[0]
            loss = criterion(logits, labels) + self.fam_lambda * reg
        else:
            loss = criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss