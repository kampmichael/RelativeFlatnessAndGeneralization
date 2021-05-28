import torch
from torch.autograd import grad, Variable
import numpy as np
from numpy import linalg as LA
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeClassifier
from copy import deepcopy
from contextlib import contextmanager

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_loss_on_data(model, loss, x, y):
    output = model(x)
    loss_value = loss(output, y)
    return output, np.sum(loss_value.data.cpu().numpy())

def softmax_accuracy(output, y):
    labels_np = y.data.cpu().numpy()
    output_np = output.data.cpu().numpy()
    acc = 0
    for i in range(len(labels_np)):
        if labels_np[i] == output_np[i].argmax():
            acc += 1
    return acc

def calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha, normalize = False):
    shape = feature_layer.shape

    layer_jacobian = grad(train_loss, feature_layer, create_graph=True, retain_graph=True)
    layer_jacobian_out = layer_jacobian[0]
    drv2 = Variable(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).cuda()
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = grad(n_grd[neuron_j].cuda(), feature_layer, retain_graph=True)[0].cuda()
    print("got hessian")

    trace_neuron_measure = 0.0
    maxeigen_neuron_measure = 0.0
    for neuron_i in range(shape[0]):
        neuron_i_weights = feature_layer[neuron_i, :].data.cpu().numpy()
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :].data.cpu().numpy()
            hessian = drv2[:,neuron_j,neuron_i,:]
            trace = np.trace(hessian.data.cpu().numpy())
            if normalize:
                trace /= 1.0*hessian.shape[0]
            trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * trace
            if neuron_j == neuron_i:
                eigenvalues = LA.eigvalsh(hessian.data.cpu().numpy())
                maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * eigenvalues[-1]
                # adding regularization term
                if alpha:
                    trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha
                    maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha

    return trace_neuron_measure, maxeigen_neuron_measure

# adapted from https://github.com/nitarshan/robust-generalization-measures/blob/master/data/generation/measures.py
@torch.no_grad()
def get_weights_only(model):
    blacklist = {'bias', 'bn'}
    return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]

@torch.no_grad()
def get_vec_params(weights):
    return torch.cat([p.view(-1) for p in weights], dim=0)

@torch.no_grad()
@contextmanager
def perturbed_model(
  model,
  sigma: float,
  rng,
  magnitude_eps = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model

@torch.no_grad()
def pacbayes_sigma(
  model,
  dataloader,
  accuracy: float,
  seed: int,
  magnitude_eps = None,
  search_depth: int = 15,
  montecarlo_samples: int = 3, #10,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
) -> float:
    lower, upper = 0, 2
    sigma = 1

    BIG_NUMBER = 10348628753
    device = next(model.parameters()).device
    rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
    rng.manual_seed(BIG_NUMBER + seed)

    for _ in range(search_depth):
        sigma = (lower + upper) / 2.0
        accuracy_samples = []
        for _ in range(montecarlo_samples):
            with perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
                loss_estimate = 0
                for data, target in dataloader:
                    logits = p_model(data.to(device))
                    pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
                    batch_correct = pred.eq(target.to(device).data.view_as(pred)).type(torch.FloatTensor).cpu()
                    loss_estimate += batch_correct.sum()
                loss_estimate /= len(dataloader.dataset)
                accuracy_samples.append(loss_estimate)
        displacement = abs(np.mean(accuracy_samples) - accuracy)
        if abs(displacement - accuracy_displacement) < displacement_tolerance:
            break
        elif displacement > accuracy_displacement:
            # Too much perturbation
            upper = sigma
        else:
            # Not perturbed enough to reach target displacement
            lower = sigma
    return sigma
#--------------------------------------------------------------------------
