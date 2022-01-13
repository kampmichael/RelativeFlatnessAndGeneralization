from utils import *
import torch
import torch.nn as nn


def computeFlatnessMeasure(model, Xorig_train, yorig_train, Xorig_test, yorig_test, with_acc = True, verbose = False):
    set_seeds(42)
    alpha = None
    
    feature_layer_id = model.feature_layer_id
    
    yorig_train = yorig_train.reshape(-1,1)
    yorig_test = yorig_test.reshape(-1,1)
    
    element_loss = nn.MSELoss()
    avg_loss = torch.nn.MSELoss()
    

    x_train = torch.Tensor(Xorig_train)
    y_train = torch.Tensor(yorig_train)
    x_test = torch.Tensor(Xorig_test)
    y_test = torch.Tensor(yorig_test)
    train_size = len(x_train)
    test_size = len(x_test)

    test_output, test_loss_overall = calculate_loss_on_data(model, element_loss, x_test, y_test)
    test_loss_overall *= (train_size//test_size)
    test_loss = avg_loss(test_output, y_test)
    if verbose:
        print("Test loss calculated", test_loss)
    if with_acc:
        acc = score_accuracy(test_output, y_test)
        if verbose:
            print("Test accuracy is", acc)

    ## train data
    train_output, train_loss_overall = calculate_loss_on_data(model, element_loss, x_train, y_train)
    if verbose:
        print("Train loss calculated", train_loss_overall)
    if with_acc:
        acc = score_accuracy(train_output, y_train)
        if verbose:
            print("Train accuracy is", acc)

    train_loss = avg_loss(train_output, y_train)

    # normalization of feature layer
    activation = model.feature_layer(x_train).data.cpu().numpy()
    activation = np.squeeze(activation)
    sigma = np.var(activation, axis=0)
    j = 0
    for p in model.parameters():
        if feature_layer_id - 2 == j or feature_layer_id - 1 == j:
            for i, sigma_i in enumerate(sigma):
                p.data[i] /= sigma_i
        if feature_layer_id == j:
            for i, sigma_i in enumerate(sigma):
                #p.data[:,i] *= sigma_i
                p.data[:,i] *= sigma_i
            feature_layer = p
        j += 1

    calculateHessianValues_fc_layer
    trace_neuron_measure, maxeigen_neuron_measure = calculateRelativeFlatnessNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha)
    trace_measure, relativeFlatness = calculateRelativeAndClassicalFlatnessTraceHessian_fc_layer(feature_layer, train_loss)
    if verbose:
        print("Neuronwise tracial measure is", trace_neuron_measure)
        print("Neuronwise max eigenvalue measure is", maxeigen_neuron_measure)
    
        print("# test loss \t train loss \t neuron_flatness_trace \t neuron_flatness_eigenvalue \n")
        print(str(test_loss_overall) + "\t" + str(train_loss_overall) + "\t" + str(trace_neuron_measure) + "\t" + str(maxeigen_neuron_measure) + "\n")
    #TODO: test_loss_overall seems to be the wrong loss measure, at least I need the MSE here. Probably, we want the loss function to be a parameter, though.
    del model
    del test_output
    del train_output
    return trace_neuron_measure, maxeigen_neuron_measure, trace_measure, relativeFlatness, test_loss.item(), train_loss.item()