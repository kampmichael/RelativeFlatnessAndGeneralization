from KDEpy import NaiveKDE
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import RidgeClassifier
RNG = default_rng(42)

def getWeights(Xphi, delta):
    return delta*np.sum(Xphi**2, axis=-1)**(1/2)

def getKernelValue(x, i, kde):
    weight = (1 / kde.data.shape[0]) if kde.weights is None else kde.weights[i]
    dist = np.sqrt(np.linalg.norm(x - kde.data[i]))
    return weight * kde.kernel(dist, bw=kde.bw, norm=kde.norm)
    
def approximateIntViaVKDE(Xphi, y, psi, delta, loss, cv = 10, useVariableKDE = True):
    '''
    This function approximates the interpolatingness of a representation for the data distribution
    by computing a vKDE on a subset of the data and calculate its fitting error on the rest. This
    serves as an estimate for how well the vKDE approximates the data distribution from the training
    set. This estimate is then used to approximate interpolatingness by calculating the loss of
    the vKDE predictions.
    
    Parameters
    ----------
    Xphi : ndarray of shape (n,m)
        The n inputs of the training dataset for which the interpolatingness should be approximated in
        the m-dimensional representation phi(X).
    y : ndarray of shape (n,)
        The labels of the training dataset corresponding to the inputs Xphi.
    psi : class
        The predictor psi that computes the prediction based on the feature representation phi(x). 
        I.e., the model f is decomposed as f = psi( phi (x) ). We assume, psi has a function
        'predict': R^m -> Y.
        
        If psi is None, then psi is optimized for each train-test split
    delta : float
        The parameter delta of the interpolatingness.
    loss : function R x R -> R
        The loss function for which interpolatingness should be approximated.
    cv : int, default = 10
        The number of cross-validation rounds used to approximate interpolatingness. 
        'None' and '0' are interpreted as performing just one round ('cv=1' ), i.e., no actual cross-validation.
    useVariableKDE : bool, default = True
        The approximation of interpolatingness requires a KDE with a bandwidth per sample x_i being 
        delta ||phi(x_i)||. This can be implemented with a vKDE with these bandwidths. However,
        we can get a more crude and inaccurate approximation by using a fixed bandwidth KDE.
        
    Returns
    -------
    IntApprox : float
        An approximation of the interpolatingness of the dataset S = (X, y) in the feature representation phi
        for a loss function loss and a predictor psi.
    '''
    
    if cv is None or cv == 0:
        cv = 1
    
    intApprox = 0.0
    n,m = Xphi.shape
    
    idxs = np.arange(n)
    RNG.shuffle(idxs)
    foldIdxs = np.array_split(idxs, cv)
    
    for i in range(cv):
        Sidx = np.concatenate([foldIdxs[j] for j in range(cv) if j != i])
        Tidx = foldIdxs[i]
        Sx, Sy, Tx, Ty =  Xphi[Sidx], y[Sidx], Xphi[Tidx], y[Tidx]
        locPsi = psi
        if psi is None:
            locPsi = RidgeClassifier()
            locPsi.fit(Sx, Sy)
        intApprox += approximateIntViaVKDE_singleRound(Sx, Sy, Tx, Ty, locPsi, delta, loss, useVariableKDE)
    intApprox /= float(cv)
    
    return intApprox
    
    
    

def approximateIntViaVKDE_singleRound(Sx, Sy, Tx, Ty, psi, delta, loss, useVariableKDE = True):
    '''
    Implements a single round of the approximation of interpolatingness used in :func:'approximateIntViaVKDE'.
    
    Parameters
    ----------
    Sx : ndarray of shape (n1,m)
        The inputs on which the vKDE is fitted.
    Sy : ndarray of shape (n1,)
        The labels to the inputs Sx.
    Tx : ndarray of shape (n2,m)
        The inputs on which the vKDE is evaluated.
    Ty : ndarray of shape (n2,)
        The labels to the inputs Tx.
    psi : function R^m -> Y
        The predictor psi that computes the prediction based on the feature representation phi(x). 
        I.e., the model f is decomposed as f = psi( phi (x) ).
    delta : float
        The parameter delta of the interpolatingness.
    loss : function R x R -> R
        The loss function for which interpolatingness should be approximated.
    useVariableKDE : bool, default = True
        The approximation of interpolatingness requires a KDE with a bandwidth per sample x_i being 
        delta ||phi(x_i)||. This can be implemented with a vKDE with these bandwidths. However,
        we can get a more crude and inaccurate approximation by using a fixed bandwidth KDE.
        
    Returns
    -------
    IntApprox : float
        An approximation of the interpolatingness of the dataset S = (X, y) in the feature representation phi
        for a loss function loss and a predictor psi.
    '''
    weights = None
    if useVariableKDE:
        weights = getWeights(Sx, delta)
    
    kde = NaiveKDE(kernel="gaussian", bw=1.)
    kde.fit(Sx, weights)
    
    sumVal = 0.0
    for j in range(len(Tx)):
        x,y = Tx[j], Ty[j]
        lossTrue = loss(y, psi.predict(x.reshape(-1, 1).T))
        
        sumLossKDE = 0.0
        for i in range(len(Sx)):
            xi, yi = Sx[i], Sy[i]
            k = getKernelValue(x, i, kde) #k(x_j,x_i)
            lossPred = loss(y, psi.predict(xi.reshape(-1, 1).T)) #l(f(x_j), y_i)
            sumLossKDE += k*lossPred
        meanLossKDE = sumLossKDE / float(len(Sx))
        
        sumVal += (lossTrue - meanLossKDE) /float(Tx.shape[0])
        
    IntApprox = sumVal #abs(sumVal) #do we need an absolute value here? -> better no absolute value...
    
    return IntApprox[0] #IntApprox is an ndarray of shape (1,) containing a single float. The function returns just the float.