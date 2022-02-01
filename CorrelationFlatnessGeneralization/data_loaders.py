from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

def load_bostonPricing(train_batch_size, test_batch_size = 100, full_data = False):
    X, y = load_boston(return_X_y=True)
    y = y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    data_scaler = StandardScaler()
    data_scaler.fit(X_train)
    scaled_train_features = data_scaler.transform(X_train)
    scaled_test_features = data_scaler.transform(X_test)

    target_scaler = StandardScaler()
    target_scaler.fit(y_train)
    scaled_train_target = target_scaler.transform(y_train)
    scaled_test_target = target_scaler.transform(y_test)

    X_train, y_train = Variable(torch.tensor(scaled_train_features).float()), Variable(torch.tensor(scaled_train_target).float())
    X_test, y_test = Variable(torch.tensor(scaled_test_features).float()), Variable(torch.tensor(scaled_test_target).float())

    if full_data:
        train_batch_size = len(X_train)
        test_batch_size = len(X_test)

    train_torch_dataset = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(dataset=train_torch_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_torch_dataset = Data.TensorDataset(X_test, y_test)
    test_loader = Data.DataLoader(dataset=test_torch_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return X.shape[1], y.shape[1], train_loader, test_loader

def load_cifar10(train_batch_size, test_batch_size = 100):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader