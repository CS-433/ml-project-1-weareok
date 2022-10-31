#%%%%%%%%%%%%%%%%%%% Higgs Boson Machine Learning%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#    crossvalidation.py includes the implementation of the cross validation for Higgs Boson Machine Learning project
#    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from helpers import *
from featureprocessing import *
from prediction import *
from implementations import *





max_iters = 10                        # GD iteration number
gamma     = np.array([0.05,0.05,0.05]) # step size for gradient descent



def cross_validation(y, x, k_indices, i, k, lambda_, method):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        i:          i-th group of data
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar
        method:     choose from "ridge_regression" and "logistic_regression"

    Returns:
        train and test logistic loss
    """
    
    iTest  = k_indices[k,:]
    iTrain = np.concatenate([k_indice for l,k_indice in enumerate(k_indices) if l!=k])
    
    xTrain = x[iTrain]
    yTrain = y[iTrain]
    xTest = x[iTest]
    yTest = y[iTest]
    total_num = np.shape(yTest)[0]
    dim = np.shape(x)[1]
    
    if method =="ridge_regression":
        w, loss_tr = ridge_regression(yTrain, xTrain, lambda_)
        loss_te    = compute_loss_MSE(yTest, xTest, w)
        y_predict  = model_prediction(xTest, w, method, 0)
        arr  =  y_predict == yTest
        correct_num  = np.sum(arr!=0)
    elif method =="logistic_regression":  
        y_label_logistic = logistic_label_mapping(yTrain)
        initial_w = np.random.random(dim)/np.sqrt(dim)
        w, loss_tr  = reg_logistic_regression(y_label_logistic, xTrain, lambda_, initial_w, max_iters, gamma[i])
        #w, loss_tr = reg_logistic_regression(y_label_logistic, xTrain, lambda_, initial_w, max_iters, gamma[i])
        loss_te    = compute_loss_MSE(yTest, xTest, w)
        y_predict  =  model_prediction(xTest, w, method, 0.5)
        arr = y_predict == yTest
        correct_num = np.sum(arr != 0)
    accuracy =   correct_num/total_num
    return w, loss_tr, loss_te, accuracy


