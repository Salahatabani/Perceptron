import numpy as np
from numpy.matlib import repmat
import sys
from scipy.io import loadmat

def loaddata(filename):
    """
    Returns xTr,yTr,xTe,yTe
    xTr, xTe are in the form nxd
    yTr, yTe are in the form nx1
    """
    data = loadmat(filename)
    xTr = data["xTr"] # load in Training data
    yTr = np.round(data["yTr"]) # load in Training labels
    xTe = data["xTe"] # load in Testing data
    yTe = np.round(data["yTe"]) # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T

def row_vectorize(x):
    return x.reshape(1,-1)

def perceptronUpdate(x,y,w):
    """
    function w=perceptronUpdate(x,y,w);
    
    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (1xd)
    y : corresponding label (-1 or +1)
    w : weight vector before updating
    
    Output:
    w : weight vector after updating
    """
    # just in case x, w are accidentally transposed (prevents future bugs)
    x,w = map(row_vectorize, [x,w])
    assert(y in {-1,1})
    ## fill in code here
    if y*np.inner(w,x)<=0:
        w = w+y*x
    return w

def perceptron(x,y,max_iter = 100):
    """
    function w=perceptron(x,y);
    
    Implementation of a Perceptron classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    
    Output:
    w : weight vector (1xd)
    """
    
    n, d = x.shape
    w = np.zeros((1,d))
    for j in range(100):
        i = np.random.permutation([i for i in range(n)])
        x = x[i,:]
        y = y[i]
        m = 0
        for k in range(n):
            w_new = perceptronUpdate(x[k,:],y[k],w)
            if np.all(w_new != w):
                m = m+1
            w = w_new
        if m ==0: break    
    return w

def classifyLinear(x,w,b=0):
    """
    function preds=classifyLinear(x,w,b)
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    w : weight vector (1xd)
    b : bias (scalar)
    
    Output:
    preds: predictions (1xn)
    """
    w = w.reshape(-1)
    #preds = np.zeros(x.shape[0])
    preds = np.array(np.sign(np.inner(w,x)+b),dtype=int)
    return preds

def polarize(x, val):
    z = np.zeros(x.shape,dtype=int)
    z[x != val] = -1
    z[x == val] = 1
    return z

if __name__ == '__main__':
    "Prepare the data and polarize the labels to 1 and -1"
    xTr,yTr,xTe,yTe=loaddata("../Dataset/digits.mat")
    yTr = yTr.astype(int)
    
    idx = np.where(np.logical_or(yTr == 0, yTr == 7).flatten())[0]
    xTr = xTr[idx,:]
    yTr = yTr[idx].flatten()
    yTr = polarize(yTr,7)
    "Train the model"
    w = perceptron(xTr,yTr)
    preds = classifyLinear(xTr,w,b=0)
    print("Accuracy on training is {0:.0f}%".format(np.mean(preds==yTr)*100))
    
    "Test the model"
    yTe = yTe.astype(int)
    idx = np.where(np.logical_or(yTe == 0, yTe == 7).flatten())[0]
    xTe = xTe[idx,:]
    yTe = yTe[idx].flatten()
    yTe = polarize(yTe,7)
    preds = classifyLinear(xTe,w,b=0)
    print("Accuracy on test is {0:.0f}%".format(np.mean(preds==yTe)*100))
