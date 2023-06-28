import torch
from torch.autograd import grad
import numpy as np
from numpy import linalg as LA
import random
import functorch

from util import torch_conv_layer_to_affine

# example call: FAMreg(inputs, targets, F.cross_entropy, autogradHessian, j_layer)

def FAMreg(inputs, targets, hessian_function, norm_function, approximate=True):
    regularizer = RelativeFlatness(hessian_function, hessian_function.f_layer, norm_function)

    return regularizer(inputs, targets, approximate)

class RelativeFlatness():
    def __init__(self, hessian_function, feature_layer, norm_function):
        self.hessian_function = hessian_function
        self.feature_layer = feature_layer
        self.norm_function = norm_function

    def __call__(self, inputs, targets, approximate):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        shape = self.feature_layer.shape

        if self.norm_function == 'neuronwise':
            H = self.hessian_function(inputs, targets, device)
            regularization = torch.zeros(1, requires_grad=True).to(device)
            for neuron_i in range(shape[0]):
                neuron_i_weights = self.feature_layer[neuron_i, :]
                for neuron_j in range(shape[0]):
                    neuron_j_weights = self.feature_layer[neuron_j, :]
                    hessian = torch.tensor(H[:, neuron_j, neuron_i, :], requires_grad=True).to(device)
                    norm_trace = torch.trace(hessian) / (1.0 * shape[1])
                    regularization = regularization + torch.dot(neuron_i_weights, neuron_j_weights) * norm_trace
        elif self.norm_function == 'layerwise_trace':
            weights_norm = torch.linalg.norm(self.feature_layer)
            regularization = weights_norm * self.hessian_function.trace(inputs, targets, approximate, device)
        elif self.norm_function == 'layerwise_eigenvalue':
            weights_norm = torch.linalg.matrix_norm(self.feature_layer, dim=tuple(range(len(shape))))
            regularization = weights_norm * self.hessian_function.max_eigenvalue(inputs, targets, approximate, device)
        return regularization

class LayerHessian():
    def __init__(self, model, layer_id, loss_function, method='functorch'):
        self.model = model
        self.layer_id = layer_id
        for i, p in enumerate(self.model.named_parameters()):
            if i == layer_id:
                self.f_name = p[0]
                self.f_layer = p[1]
                break
        self.loss_function = loss_function
        self.method = method

    def __call__(self, inputs, targets, device):
        if self.method == 'functorch':
            func = lambda params: self.loss_function(
                torch.nn.utils.stateless.functional_call(self.model, {self.f_name: params}, inputs).to(device),
                targets)

            hessian = functorch.jacfwd(functorch.jacrev(func), randomness='same')(self.f_layer)
            return hessian
        elif self.method == 'autograd':
            loss = self.loss_function(self.model(inputs), targets)
            shape = self.f_layer.shape
            # need to retain_graph for further hessian calculation
            # need allow_unused for the layer that is transformed from conv2d
            layer_jacobian = grad(loss, self.f_layer, create_graph=True, retain_graph=True, allow_unused=True)
            drv2 = torch.tensor(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).to(device)
            for ind, n_grd in enumerate(layer_jacobian[0].T):
                for neuron_j in range(shape[0]):
                    drv2[ind][neuron_j] = grad(n_grd[neuron_j].to(device), self.f_layer, retain_graph=True)[0].to(device)
            return drv2
        elif self.method == 'autograd_fast':
            func = lambda params: self.loss_function(
                torch.nn.utils.stateless.functional_call(self.model, {self.f_name: params}, inputs).to(device),
                targets)

            hessian = torch.autograd.functional.hessian(func, self.f_layer, vectorize=True, create_graph=True)
            return hessian
        else:
            print("No such method of hessian computation")
        return None

    def trace(self, inputs, targets, approximate, device):
        shape = self.f_layer.shape
        if approximate:
            loss = self.loss_function(self.model(inputs), targets)
            grads = [grad(loss, self.f_layer, create_graph=True, retain_graph=True)[0]]
            #loss.backward(create_graph=True)

            #for i, param in enumerate(self.model.parameters()):
            #    #if not param.requires_grad:
            #    #    continue
            #    if i == self.layer_id:
            #        grads = [0. if param.grad is None else param.grad + 0.]
            #        break

            tol = 1e-4
            trace_vhv = []
            trace = 0.

            for i in range(200):
                self.model.zero_grad()
                v = [torch.randint_like(p, high=2, device=device) for p in [self.f_layer]]

                # generate Rademacher random variables
                for v_i in v:
                    v_i[v_i == 0] = -1
                Hv = torch.autograd.grad(grads, [self.f_layer], grad_outputs=v, create_graph=True)
                trace_vhv.append(sum([torch.sum(x * y) for (x, y) in zip(Hv, v)]))
                if abs(torch.mean(torch.tensor(trace_vhv)) - trace) / (abs(trace) + 1e-6) < tol:
                    return torch.mean(torch.tensor(trace_vhv, requires_grad=True))
                else:
                    trace = torch.mean(torch.tensor(trace_vhv, requires_grad=True))

            return torch.mean(torch.tensor(trace_vhv, requires_grad=True))
        else:
            return torch.trace(self(inputs, targets, device).reshape(np.prod(shape), np.prod(shape)))

    def max_eigenvalue(self, inputs, targets, approximate, device):
        shape = self.f_layer.shape
        if approximate:
            return None
        else:
            return torch.lobpcg(self(inputs, targets, device).reshape(np.prod(shape), np.prod(shape)))


'''
def FAMloss(output, target, loss_function, feature_layer, lmb=0.01):
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_part = loss_function(output, target)

    regularization = Variable(torch.zeros(1), requires_grad=True).to(device)
    shape = feature_layer.shape
    layer_jacobian = grad(loss_part, feature_layer, create_graph=True, retain_graph=True)
    drv2 = Variable(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).to(device)
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = grad(n_grd[neuron_j].to(device), feature_layer, retain_graph=True)[0].to(device)

    for neuron_i in range(shape[0]):
        neuron_i_weights = feature_layer[neuron_i, :]
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :]
            hessian = Variable(torch.tensor(drv2[:, neuron_j, neuron_i, :]), requires_grad=True).to(device)
            norm_trace = torch.trace(hessian) / (1.0 * shape[1])
            regularization = regularization + torch.dot(neuron_i_weights, neuron_j_weights) * norm_trace

    end = time.time()
    print("Elapsed time", end - start)

    print(regularization)
    return loss_part + lmb*regularization
'''

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


