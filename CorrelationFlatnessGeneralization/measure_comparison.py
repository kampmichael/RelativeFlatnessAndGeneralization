import torch
import os
import math
import numpy as np

from utils import *
from data_loaders import *

#parameters to set
feature_layer_id = 8
reparam_layers = [6,7]
nets_dir = 'loss_less_01'
netw = "LeNet5"
element_loss = torch.nn.CrossEntropyLoss(reduction='none')
avg_loss = torch.nn.CrossEntropyLoss()
with_acc = True
filename = nets_dir + "/measures/reparametrized_comparison_measures_"
#//

set_seeds(1)
trainloader, testloader = load_cifar10(train_batch_size = 50000, test_batch_size = 10000)
input_dim = 32*32*3
output_dim = 10
inputs, labels = iter(trainloader).next()
x_train = inputs.cuda()
y_train = labels.cuda()
inputs, labels = iter(testloader).next()
x_test = inputs.cuda()
y_test = labels.cuda()
train_size = len(x_train)
test_size = len(x_test)
print("Data loaded", train_size, "training", test_size, "testing")

for trained_net in os.listdir(nets_dir):
    if trained_net=='measures':
        continue
    print("---------------", trained_net)

    if 'wd' in trained_net:
        alpha = float(trained_net.split('wd')[1].split('_')[0])
        print("Weight decay with", alpha)
    else:
        alpha = None
        print("No weight decay")

    np.random.seed(None)
    # factor for reparametrization
    factor = 1.0 * np.random.randint(low=5, high=26)
    print("random factor", factor)

    from models import *

    set_seeds(7)
    model = eval(netw)(input_dim = input_dim, output_dim = output_dim).cuda()    
    try:
        model.load_state_dict(torch.load(os.path.join(nets_dir, trained_net)))
    except:
        print("The weights cannot be loaded to the specified architecture; skipping")
        continue
    model.eval()
    print("Model loaded")

    set_seeds(7)
    model = eval(netw)(input_dim = input_dim, output_dim = output_dim).cuda()
    model.load_state_dict(torch.load(os.path.join(nets_dir, trained_net)))
    model.eval()

    i = 0
    for l in model.parameters():
        if i in reparam_layers:
            l.data = l.data * 1.0/ factor
        elif i == feature_layer_id:
            l.data = l.data * factor
        i += 1

    ## test data
    test_output, test_loss = calculate_loss_on_data(model, element_loss, x_test, y_test)
    test_loss *= (train_size//test_size)
    print("Test loss calculated", test_loss)
    if with_acc:
        acc = softmax_accuracy(test_output, y_test)
        print("Test accuracy is", acc*1.0/len(y_test))

    ## train data
    train_output, train_loss_overall = calculate_loss_on_data(model, element_loss, x_train, y_train)
    print("Train loss calculated", train_loss_overall)
    if with_acc:
        acc = softmax_accuracy(train_output, y_train)
        print("Train accuracy is", acc*1.0/len(y_train))
        train_acc = acc*1.0/len(y_train)

    train_loss = avg_loss(train_output, y_train)

    # hessian calculation for the layer of interest
    i = 0
    for p in model.parameters():
        if i == feature_layer_id:
            feature_layer = p
        i += 1
    last_layer_jacobian = grad(train_loss, feature_layer, create_graph=True, retain_graph=True)
    hessian = []
    for n_grd in last_layer_jacobian[0]:
        for w_grd in n_grd:
            drv2 = grad(w_grd, feature_layer, retain_graph=True)
            hessian.append(drv2[0].data.cpu().numpy().flatten())

    weights_norm = 0.0
    for n in feature_layer.data.cpu().numpy():
        for w in n:
            weights_norm += w**2
    print("Squared euclidian norm is calculated", weights_norm)

    max_eignv = LA.eigvalsh(hessian)[-1]
    print("Largest eigenvalue is", max_eignv)

    trace = np.trace(hessian)
    print("Trace is", trace)

    ## calculate FisherRao norm
    # analytical formula for crossentropy loss from Appendix of the original paper
    sum_derivatives = 0
    m = torch.nn.Softmax(dim=0)
    for inp in range(len(train_output)):
        sum_derivatives += \
            (np.inner(m(train_output[inp]).data.cpu().numpy(), train_output[inp].data.cpu().numpy()) -
             train_output[inp].data.cpu().numpy()[y_train[inp]]) ** 2
    fr_norm = math.sqrt(((5 + 1) ** 2) * (1.0 / len(train_output)) * sum_derivatives)
    print("Fisher Rao norm is", fr_norm)

    # adapted from https://github.com/nitarshan/robust-generalization-measures/blob/master/data/generation/measures.py
    sigma = pacbayes_sigma(model, trainloader, train_acc, 42)
    weights = get_weights_only(model)
    w_vec = get_vec_params(weights)
    pacbayes_flat = 1.0 / sigma ** 2
    print("PacBayes flatness", pacbayes_flat)
    def pacbayes_bound(reference_vec):
        return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(train_size / sigma) + 10
    pacbayes_orig = pacbayes_bound(w_vec).data.cpu().item()
    print("PacBayes orig", pacbayes_orig)
    #-----------------------------

    # normalization of feature layer
    activation = model.feature_layer(x_train).data.cpu().numpy()
    activation = np.squeeze(activation)
    sigma = np.std(activation, axis=0)

    j = 0
    for p in model.parameters():
        if feature_layer_id - 2 == j or feature_layer_id - 1 == j:
            for i, sigma_i in enumerate(sigma):
                if sigma_i != 0.0:
                    p.data[i] = p.data[i] / sigma_i
        if feature_layer_id == j:
            for i, sigma_i in enumerate(sigma):
                p.data[:,i] = p.data[:,i] * sigma_i
            feature_layer = p
        j += 1
        
    train_output, train_loss_overall = calculate_loss_on_data(model, element_loss, x_train, y_train)
    train_loss = avg_loss(train_output, y_train)

    trace_nm, maxeigen_nm = calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha, normalize = False)
    print("Neuronwise tracial measure is", trace_nm)
    print("Neuronwise max eigenvalue measure is", maxeigen_nm)

    with open(filename + trained_net + ".txt", "w") as outp:
        outp.write(
            "# train_loss \t test_loss \t weights_norm \t max_eignv \t trace \t fisher_rao \t pacbayes_fl \t pacbayes_orig \t trace_m \t meig_m \n")
        outp.write(str(train_loss_overall) + "\t" + str(test_loss) + "\t" + str(weights_norm) + "\t" + str(max_eignv) + "\t" + str(trace) +
            "\t" + str(fr_norm) + "\t" + str(pacbayes_flat) + "\t" + str(pacbayes_orig) + "\t" + str(trace_nm) + "\t" + str(maxeigen_nm) + "\n")

    del model
    del test_output
    del train_output
    del train_loss_overall
    torch.cuda.empty_cache()
