#%%%%%%%%%%%%%%%%%%% Higgs Boson Machine Learning%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#    prediction.py file includes the code implementation to predict the test set
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from implementations import *

def model_prediction(x_output,w_start, method, decision_bound):

    output = np.dot(x_output,w_start) 
    N = x_output.shape[0]
    label_prediction  = np.zeros(N)   # store the results
    
    if method == "ridge_regression":
        indx_positive = output>=decision_bound
        indx_negative = output<decision_bound        
        label_prediction[indx_positive]  = int(1)
        label_prediction[indx_negative]  = int(-1)
    elif method == "logistic_regression":
        sigmoid_vec = sigmoid(output)
        indx_positive = output>=decision_bound
        indx_negative = output<decision_bound 
        label_prediction[indx_positive]  = int(1)
        label_prediction[indx_negative]  = int(-1)
        
    return label_prediction