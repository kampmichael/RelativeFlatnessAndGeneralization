import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class simpleMLP(nn.Module):
    def __init__(self, features):
        self.m = features
        super(simpleMLP,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,features)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class MLPandRidge2D(nn.Module):
    def __init__(self, features = 2):
        self.m = features
        super(MLPandRidge2D,self).__init__()
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,features)
        self.ridge = nn.Linear(features,1)
        
        self.feature_layer_id = 6
        
    def forward(self,x):
        x = x.view(-1,2)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.ridge(x)
        return x
    
    def feature_layer(self, x): #this one is needed for the flatness computation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x.view(-1, self.m, 1)

class MLPandRidge(nn.Module):
    def __init__(self, features = 2):
        self.m = features
        super(MLPandRidge,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,features)
        self.ridge = nn.Linear(features,1)
        
        self.feature_layer_id = 6
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.ridge(x)
        return x
    
    def feature_layer(self, x): #this one is needed for the flatness computation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x.view(-1, self.m, 1)
        
        
class simpleDeepMLP(nn.Module):
    def __init__(self, features):
        self.m = features
        super(simpleDeepMLP,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,384)
        self.fc3 = nn.Linear(384,256)
        self.fc4 = nn.Linear(256,196)
        self.fc5 = nn.Linear(196,128)
        self.fc6 = nn.Linear(128,32)
        self.fc7 = nn.Linear(32,features)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.xavier_uniform_(self.fc7.weight)
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x)
        return x
        
class DeepMLPandRidge(nn.Module):
    def __init__(self, features = 2):
        self.m = features
        super(DeepMLPandRidge,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,384)
        self.fc3 = nn.Linear(384,256)
        self.fc4 = nn.Linear(256,196)
        self.fc5 = nn.Linear(196,128)
        self.fc6 = nn.Linear(128,32)
        self.fc7 = nn.Linear(32,features)
        self.ridge = nn.Linear(features,1)
        
        self.feature_layer_id = 12
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = self.ridge(x)
        return x
    
    def feature_layer(self, x): #this one is needed for the flatness computation
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        return x.view(-1, self.m, 1)
        
class simpleLinearMLP(nn.Module):
    def __init__(self, features):
        self.m = features
        super(simpleLinearMLP,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,features)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class LinearMLPandRidge(nn.Module):
    def __init__(self, features = 2):
        self.m = features
        super(LinearMLPandRidge,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,features)
        self.ridge = nn.Linear(features,1)
        
        self.feature_layer_id = 6
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.ridge(x)
        return x
    
    def feature_layer(self, x): #this one is needed for the flatness computation
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.view(-1, self.m, 1)
        
def reverseModelPropagation(model, y):
    z = torch.tensor(y).float()  # use 'z' for the reverse result, start with the model's output 'y'.
    for step in list(model.children())[::-1]:
        if isinstance(step, nn.Linear):
            #we need to solve Wx + b = z for x, i.e., Wx = z-b
            z = z - step.bias[None, ...] #first, compute z-b      
            z = torch.lstsq(z.t(), step.weight)[0].t() #computes the least-squares solution min_x||Wx + b - z||^2_2
        elif isinstance(step, torch.tanh):
            #this is simply the inverse of the activation function for tanh
            z = 0.5 * torch.log((1 + z) / (1 - z))
    return z
    
def testReverseModel(model):
    N = 100  # number of samples
    n = 3   # number of neurons per layer

    x = torch.randn(N, 28*28)
    y = model.forward(x)
    z = reverseModelPropagation(model, y)

    y2 = model.forward(z)

    print(torch.dist(y,y2))

    y = y.detach().numpy()
    y2 = y2.detach().numpy()

    plt.scatter(y[:,0], y[:,1], label="true labels")
    plt.scatter(y2[:,0], y2[:,1], label="labels of reconstructed input")
    plt.legend()
    plt.show()
    
def combineWeights(f, mlp, ridge):
    new_state_dict = f.state_dict()
    for key in mlp.state_dict():
        new_state_dict[key] = mlp.state_dict()[key]
    new_state_dict['ridge.weight'] = torch.tensor(ridge.coef_)
    new_state_dict['ridge.bias'] = torch.tensor(ridge.intercept_)
    f.load_state_dict(new_state_dict)
    return f

def getReverseEngineeredModel_ridge(psi_ridge):
    m = 0
    m = psi_ridge.coef_.shape[1]
    phi = simpleMLP(features = m)
    f = MLPandRidge(features = m)
    f = combineWeights(f, phi, psi_ridge)
    return f, phi
    
def getReverseEngineeredModel_DeepRidge(psi_ridge):
    m = 0
    m = psi_ridge.coef_.shape[1]
    phi = simpleDeepMLP(features = m)
    f = DeepMLPandRidge(features = m)
    f = combineWeights(f, phi, psi_ridge)
    return f, phi
    
def getReverseEngineeredModel_LinearRidge(psi_ridge):
    m = 0
    m = psi_ridge.coef_.shape[1]
    phi = simpleLinearMLP(features = m)
    f = LinearMLPandRidge(features = m)
    f = combineWeights(f, phi, psi_ridge)
    return f, phi