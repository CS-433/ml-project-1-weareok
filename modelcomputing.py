#%%%%%%%%%%%%%%%%%%% Higgs Boson Machine Learning%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#    modelcomputing.py includes training code for Higgs Boson Machine Learning project, it includes: 
#    1) data processing, 2) Model estimation
#    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy
import numpy as np
from preprocessing import *
from helpers import *
from crossvalidation import *

print("Program started")
lambdas = np.array([0.0001])       # regularization strength
k_fold  = 30                       # cross validation fold

# define the method to train the model
method = "ridge_regression"


# load training data set 
data_path = "train.csv"
y_label, x_feature, ids = load_csv_data(data_path, sub_sample=False)
print("Training dataset is loaded")


# data preprocessing 
print("Preprocessing is started")
y_labels, x_outputs = preprocessing(y_label, x_feature)
print("Preprocessing finished")



# model estimation 
print("Starting model estimation")
ws = [np.zeros(x_outputs[0].shape[1]), np.zeros(x_outputs[1].shape[1]),np.zeros(x_outputs[2].shape[1])]
for i in range(3):      # Indexing data group, we have three
    print("Predicting Model {}".format(i))
    x = x_outputs[i]    # Get the regression matrix
    y = y_labels[i]
    dim = x.shape[1]    # Get the dimension of features
    
    k_indices = build_k_indices(y, k_fold, 0+i) # Generating index of cross-validation set

    rmse_tr, rmse_te, racc_tr = [], [], []
    for lambda_ in lambdas:  #  Indexing regularization strength 
        print("Regularization strength = {}".format(lambda_))
        losses_tr, losses_te , acc_tr = [],  [],  []
        w_temp = np.zeros((k_fold, np.shape(x)[1]))
        for k in range(k_fold):    # Indexing cross validation fold     
            w_start, loss_tr, loss_te, accuracy = cross_validation(y, x, k_indices, i, k, lambda_, method) # run cross validation
            losses_tr.append(loss_tr)
            losses_te.append(loss_te)
            acc_tr.append(accuracy)
            w_temp[k] = w_start
            print("{}-th fold of group {}:  training loss error = {},  testing loss error ={}, test accuracy = {}".format(
                k, i,loss_tr,loss_te,accuracy
            ))
        rmse_tr.append(np.mean(losses_tr))
        rmse_te.append(np.mean(losses_te))
        racc_tr.append(np.mean(acc_tr))
    ws[i] = np.mean(w_temp, axis = 0)  # model average
    print("Model{} Prediction finished".format(i))

    print("training loss value of these three models are {}".format(rmse_tr))
    print("testing loss value of these three models are {}".format(rmse_te))
    print("average test accuracy of these three models are {}".format(racc_tr))

    i_best_rmse = np.argmin(rmse_te)
    best_rmse = rmse_te[i_best_rmse]
    best_lambda = lambdas[i_best_rmse]
    print("for jet n°{} the best lambda is {}".format(i, best_lambda))


        

for i in range(3): # storage model parameter
    file_path = 'w{}.npy'.format(i)
    np.save(file_path, ws[i])



