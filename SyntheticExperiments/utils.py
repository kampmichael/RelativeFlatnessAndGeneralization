import torch
from torch.autograd import grad
import numpy as np
from numpy import linalg as LA
import random
from sklearn.metrics import accuracy_score
from sklearn.datasets._samples_generator import _generate_hypercube
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state, shuffle

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
	
def score_accuracy(output, y):
    labels_np = y.data.cpu().numpy()
    scores_np = output.data.cpu().numpy()
    output_np = np.where(scores_np > 0.0, 1.0, 0.0)
    return accuracy_score(labels_np, output_np)

class ClassSeps:
    def __init__(self, class_sep_list):
        self.classSepList = class_sep_list
       
    def __iter__(self):
        self.i = -1
        return self
    
    def __next__(self):
        if self.i + 1< len(self.classSepList):
            self.i += 1
            nextElem = self.classSepList[self.i]
            return nextElem
        else:
            raise StopIteration
            
    def __call__(self):
        return self.classSepList[-1*(self.i+1)]
        
    

def getClassSepDatasets(n_samples, n_features, n_informative, n_redundant, class_sep_list, clusters = 4, classes = 2, seed = None):
    datasets = {}
    cs = ClassSeps(class_sep_list)
    for class_sep in cs:
        #print(class_sep, c) n_samples=5000, n_features=8, n_informative=6, n_redundant=2, class_sep=c
        #X,y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, class_sep=cs())
        #datasets[class_sep] = (X, y)
        #continue
        samplesPerCluster = [int(n_samples * ([1.0 / classes] * classes)[k % classes] / (clusters/classes)) for k in range(clusters)]
        overallSamples = sum(samplesPerCluster)
        for i in range(n_samples - overallSamples):
            samplesPerCluster[i % clusters] += 1
        X, y = np.zeros((n_samples, n_features)), np.zeros(n_samples, dtype=int)
        rng = check_random_state(seed)
        centroids = _generate_hypercube(clusters, n_informative, rng).astype(float, copy=False)
        c = cs()
        centroids *= 2 * c
        centroids -= c
        X[:, :n_informative] = rng.randn(n_samples, n_informative)
        stop = 0
        for k, centroid in enumerate(centroids):
            start, stop = stop, stop + samplesPerCluster[k]
            y[start:stop] = k % classes  
            X_k = X[start:stop, :n_informative] 
            A = 1. * rng.rand(n_informative, n_informative) - 1
            X_k[...] = np.dot(X_k, A)  
            X_k += centroid 
        if n_redundant > 0:
            B = 1. * rng.rand(n_informative, n_redundant) - 1
            X[:, n_informative:n_informative + n_redundant] = \
                np.dot(X[:, :n_informative], B)
        if n_features - n_informative - n_redundant > 0:
            X[:, -(n_features - n_informative - n_redundant):] = rng.randn(n_samples, (n_features - n_informative - n_redundant))
        flip_mask = rng.rand(n_samples) < 0.001
        y[flip_mask] = rng.randint(classes, size=flip_mask.sum())
        X, y = shuffle(X, y, random_state=rng)
        indices = np.arange(n_features)
        rng.shuffle(indices)
        X[:, :] = X[:, indices]
        datasets[class_sep] = (X, y)
    return datasets
    
def getClassSepDataset(n_samples, n_features, n_informative, n_redundant, class_sep):
    clusters = 4
    classes = 2
    samplesPerCluster = [int(n_samples * ([1.0 / classes] * classes)[k % classes] / (clusters/classes)) for k in range(clusters)]
    overallSamples = sum(samplesPerCluster)
    for i in range(n_samples - overallSamples):
        samplesPerCluster[i % clusters] += 1
    X, y = np.zeros((n_samples, n_features)), np.zeros(n_samples, dtype=int)
    rng = check_random_state(None)
    centroids = _generate_hypercube(clusters, n_informative, rng).astype(float, copy=False)
    centroids *= 2 * class_sep
    centroids -= class_sep
    X[:, :n_informative] = rng.randn(n_samples, n_informative)
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + samplesPerCluster[k]
        y[start:stop] = k % classes  
        X_k = X[start:stop, :n_informative]  
        A = 2. * rng.rand(n_informative, n_informative) - 1
        X_k[...] = np.dot(X_k, A)  
        X_k += centroid  
    if n_redundant > 0:
        B = 2. * rng.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)
    if n_features - n_informative - n_redundant > 0:
        X[:, -(n_features - n_informative - n_redundant):] = rng.randn(n_samples, (n_features - n_informative - n_redundant))
    flip_mask = rng.rand(n_samples) < 0.001
    y[flip_mask] = rng.randint(classes, size=flip_mask.sum())
    X, y = shuffle(X, y, random_state=rng)
    indices = np.arange(n_features)
    rng.shuffle(indices)
    X[:, :] = X[:, indices]
    return X, y
 
def calculateHessianValues_fc_layer(feature_layer, train_loss, verbose=False):
    # instead of using loss.backward(), use torch.autograd.grad() to compute gradients
    feature_layer_jacobian = grad(train_loss, feature_layer, create_graph=True, retain_graph=True)
    hessian = []
    for n_grd in feature_layer_jacobian[0]:
        for w_grd in n_grd:
            drv2 = grad(w_grd, feature_layer, retain_graph=True)
            hessian.append(drv2[0].data.cpu().numpy().flatten())
    #last_layer_bias_jacobian = grad(train_loss, feature_layer_bias, create_graph=True, retain_graph=True)
    #hessian_bias = []
    #for b_grd in last_layer_bias_jacobian[0]:
    #    drv2_bias = grad(b_grd, feature_layer_bias, retain_graph=True)
    #    hessian_bias.append(drv2_bias[0].data.cpu().numpy().flatten())
    if verbose:
        print(hessian[0].shape)
        print("Hessian calculated")

    ## calculate the largest eigenvalue
    eigenvalues = LA.eigvalsh(hessian)
    max_eignv = eigenvalues[-1]
    #print("largest eigenvalue is", max_eignv)

    ## calculate normalized trace
    trace = np.trace(hessian)
    norm_trace = trace / (1.0*len(hessian))
    if verbose:
        print("normalized trace is", norm_trace)

    return max_eignv, norm_trace
    
def calculateRelativeAndClassicalFlatnessTraceHessian_fc_layer(feature_layer, train_loss):
    max_eignv, norm_trace = calculateHessianValues_fc_layer(feature_layer, train_loss)
    weights_norm = 0.0
    for n in feature_layer.data.cpu().numpy():
        for w in n:
            weights_norm += w**2
    return norm_trace, norm_trace*weights_norm

def calculateRelativeFlatnessNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha, verbose = False):
	# 10,84
    shape = feature_layer.shape
    layer_jacobian = grad(train_loss, feature_layer, create_graph=True, retain_graph=True)
    layer_jacobian_out = layer_jacobian[0]
    # 10,84
    if verbose:
        print(layer_jacobian_out.shape)

    trace_neuron_measure = 0.0
    maxeigen_neuron_measure = 0.0
    # 10
    for neuron_i in range(shape[0]):
        hessians = []
        # 84 times arrays of 10
        for ind, n_grd in enumerate(layer_jacobian_out.T):
            for neuron_j in range(shape[0]):
                if ind == 0:
                    hessians.append([])
                drv2 = grad(n_grd[neuron_j], feature_layer, retain_graph=True)
                hessians[neuron_j].append(drv2[0][neuron_i, :].data.cpu().numpy().tolist())

        hessians = [np.array(e) for e in hessians]
        if verbose:
            print("{} hessians number {} of size {} calculated".format(len(hessians), neuron_i, hessians[0].shape))

        # 84
        neuron_i_weights = feature_layer[neuron_i, :].data.cpu().numpy()
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :].data.cpu().numpy()
            norm_trace = np.trace(hessians[neuron_j]) / (1.0*hessians[neuron_j].shape[0])
            trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * norm_trace
            if neuron_j == neuron_i:
                eigenvalues = LA.eigvalsh(hessians[neuron_j])
                maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * eigenvalues[-1]
                # adding regularization term
                if alpha:
                    trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha
                    maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha

    return trace_neuron_measure, maxeigen_neuron_measure