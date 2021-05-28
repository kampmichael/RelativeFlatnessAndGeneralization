import torch

from data_loaders import load_cifar10
from utils import *

#parameters to set
BS = 64
LR = 0.005
WD = 5e-4
layer_id = 8 # gradient stopping criterion
epsilon = 1.0e-5 # gradient stopping criterion
netw = "LeNet5"
exps_count = 10
filename = "loss_less_01/cifar10_adam_lenet5_bs" + str(BS) + "_lr" + str(LR) + "_wd" + str(WD)
#//

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for the same data shuffling on every run
set_seeds(1)
trainloader, testloader = load_cifar10(train_batch_size = BS)

for exp in range(exps_count):    
    set_seeds(exp)

    from models import *

    net = eval(netw)(output_dim = 10)
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=WD)
    #optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=LR)

    epoch = 0
    train = True
    while train:
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        #grads = None
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            ## gradient size stopping criterion
            #i = 0
            #for p in net.parameters():
            #    if i == layer_id:
            #        if grads is None:
            #            grads = p.grad.data.cpu().numpy()
            #        else:
            #            grads += p.grad.data.cpu().numpy()
            #    i += 1

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        ## gradient size stopping criterion
        #all_grads_small = True
        #for g in grads.flatten():
        #    all_grads_small = all_grads_small and (abs(g/(batch_idx+1)) <= epsilon)
        #if all_grads_small:
        #    train = False

        epoch_loss = train_loss/(batch_idx+1)
        if epoch_loss < 0.1:
            train = False
        #if epoch == 150:
        #    train = False
        #if correct == total:
        #    train = False

        print('Loss: %.10f | Acc: %.3f%% (%d/%d)' % (epoch_loss, 100.*correct/total, correct, total))
        epoch += 1

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.10f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    torch.save(net.state_dict(), filename + "_exp" + str(exp))
