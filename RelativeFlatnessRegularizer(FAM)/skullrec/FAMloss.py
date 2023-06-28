import torch
from torch.autograd import grad
import numpy as np
import functorch

# example call: FAMreg(inputs, targets, F.cross_entropy, autogradHessian, j_layer)

def FAMreg(inputs, outputs, targets, hessian_function, norm_function, approximate=True):
    regularizer = RelativeFlatness(hessian_function, hessian_function.f_layer, norm_function)

    return regularizer(inputs, outputs, targets, approximate)

class RelativeFlatness():
    def __init__(self, hessian_function, feature_layer, norm_function):
        self.hessian_function = hessian_function
        self.feature_layer = feature_layer
        self.norm_function = norm_function

    def __call__(self, inputs, outputs, targets, approximate):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        shape = self.feature_layer.shape

        if self.norm_function == 'neuronwise':
            H = self.hessian_function(inputs, outputs, targets, device)
            regularization = torch.zeros(1, requires_grad=True).to(device)
            for neuron_i in range(shape[0]):
                neuron_i_weights = self.feature_layer[neuron_i, :]
                for neuron_j in range(shape[0]):
                    neuron_j_weights = self.feature_layer[neuron_j, :]
                    hessian = torch.tensor(H[:, neuron_j, neuron_i, :], requires_grad=True).to(device)
                    norm_trace = torch.trace(hessian) / (1.0 * shape[1])
                    regularization = regularization + torch.dot(neuron_i_weights, neuron_j_weights) * norm_trace
        elif self.norm_function == 'layerwise_trace':
            weights_norm = torch.linalg.matrix_norm(self.feature_layer, dim=tuple(range(len(shape))))
            regularization = weights_norm * self.hessian_function.trace(inputs, outputs, targets, approximate, device)
        elif self.norm_function == 'layerwise_eigenvalue':
            weights_norm = torch.linalg.matrix_norm(self.feature_layer, dim=tuple(range(len(shape))))
            regularization = weights_norm * self.hessian_function.max_eigenvalue(inputs, outputs, targets, approximate, device)
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

    def __call__(self, inputs, outputs, targets, device):
        if self.method == 'functorch':
            func = lambda params: self.loss_function(
                torch.nn.utils.stateless.functional_call(self.model, {self.f_name: params}, inputs).to(device),
                targets)

            hessian = functorch.jacfwd(functorch.jacrev(func), randomness='same')(self.f_layer)
            return hessian
        elif self.method == 'autograd':
            loss = self.loss_function(outputs, targets)
            shape = self.f_layer.shape
            # need to retain_graph for further hessian calculation
            layer_jacobian = grad(loss, self.f_layer, create_graph=True, retain_graph=True, allow_unused=True)
            drv2 = torch.tensor(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).to(device)
            for ind, n_grd in enumerate(layer_jacobian[0].T):
                for neuron_j in range(shape[0]):
                    drv2[ind][neuron_j] = grad(n_grd[neuron_j].to(device), self.f_layer, retain_graph=True)[0].to(device)
            return drv2
        else:
            print("No such method of hessian computation")
        return None

    def trace(self, inputs, outputs, targets, approximate, device):
        shape = self.f_layer.shape
        if approximate:
            loss = self.loss_function(outputs, targets)
            grads = [grad(loss, self.f_layer, create_graph=True, retain_graph=True)[0]]

            tol = 1e-4
            trace_vhv = []
            trace = 0.

            for i in range(200):
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
            return torch.trace(self(inputs, outputs, targets, device).reshape(np.prod(shape), np.prod(shape)))

    def max_eigenvalue(self, inputs, outputs, targets, approximate, device):
        shape = self.f_layer.shape
        if approximate:
            return None
        else:
            return torch.lobpcg(self(inputs, outputs, targets, device).reshape(np.prod(shape), np.prod(shape)))