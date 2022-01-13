import torch
import os

from utils import *
from data_loaders import *

#parameters to set
#feature_layer_id = 8
feature_layer_id = 6
#nets_dir = 'equal_epochs_500' 
#nets_dir = 'loss_less_01'
nets_dir = 'regression/loss_less_005'
#netw = "LeNet5"
netw = "DenseNet"
#element_loss = torch.nn.CrossEntropyLoss(reduction='none')
element_loss = torch.nn.MSELoss(reduction='none')
#avg_loss = torch.nn.CrossEntropyLoss()
avg_loss = torch.nn.MSELoss()
with_acc = False
#filename = nets_dir + "/measures/neuronwise_flatness_"
filename = nets_dir + "/measures/normalized_neuronwise_flatness_"
#//

set_seeds(1)
#trainloader, testloader = load_cifar10(train_batch_size = train_size, test_batch_size = test_size)
#output_dim = 10
input_dim, output_dim, trainloader, testloader = load_bostonPricing(train_batch_size = 0, full_data = True)
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

    from models import *

    set_seeds(7)
    model = eval(netw)(input_dim = input_dim, output_dim = output_dim).cuda()
    model.load_state_dict(torch.load(os.path.join(nets_dir, trained_net)))
    model.eval()
    print("Model loaded")

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
                p.data[:,i] *= sigma_i
            feature_layer = p
        j += 1

    trace_neuron_measure, maxeigen_neuron_measure = calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha)
    print("Neuronwise tracial measure is", trace_neuron_measure)
    print("Neuronwise max eigenvalue measure is", maxeigen_neuron_measure)

    with open(filename + trained_net + ".txt", "w") as outp:
        outp.write(
            "# test loss \t train loss \t neuron_flatness_trace \t neuron_flatness_eigenvalue \n")
        outp.write(str(test_loss) + "\t" + str(train_loss_overall) + "\t" + str(trace_neuron_measure) +
                   "\t" + str(maxeigen_neuron_measure) + "\n")

    del model
    del test_output
    del train_output
    torch.cuda.empty_cache()
