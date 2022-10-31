#%%%%%%%%%%%%%%%%%%% Higgs Boson Machine Learning%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#    run.py file is the testing code, used to generate the exact simulation results as submitted to
#    AIcrowd, which achieves 83.7% accuracy and 0.750 F1 score. We should first use
#    modelcomputing.py to train the model before running run.py. Nonetheless, a pretrained models
#    are also ready for testing 
#
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
from helpers import *
from featureprocessing import *
from preprocessing import *
from prediction import *



print("Programm started")
method = "ridge_regression"


data_path = "test.csv"
y_label, x_feature, ids = load_csv_data(data_path, sub_sample=False)
print("Test data loaded successfully")


# load the pre-estimated model parameter
print(method)
file_path = 'w0.npy'
w0 = np.load(file_path)
file_path = 'w1.npy'
w1 = np.load(file_path)
file_path = 'w2.npy'
w2= np.load(file_path)

# data preprocessing
print("Preprocessing is started")
y_labels, x_outputs = preprocessing(y_label, x_feature)
print("Preprocessing finished")

x_output0 = x_outputs[0]
x_output1 = x_outputs[1]
x_output2 = x_outputs[2]

print("Starting to predict")  
if method == "ridge_regression":
    predict0 = model_prediction(x_output0, w0, method, decision_bound = 0)
    predict1 = model_prediction(x_output1, w1, method, decision_bound = 0)
    predict2 = model_prediction(x_output2, w2, method, decision_bound = 0)
elif method == "logistic_regression": 
    predict0 = model_prediction(x_output0, w0, method, decision_bound = 0.5)
    predict1 = model_prediction(x_output1, w1, method, decision_bound = 0.5)
    predict2 = model_prediction(x_output2, w2, method, decision_bound = 0.5)
print("Prediction finished")  

N  = np.shape(y_label)[0]
y_pred = np.zeros((N, 1))
jet_ind = x_feature[:,22]

y_pred[jet_ind ==0] = predict0.reshape((len(predict0),1))
y_pred[jet_ind ==1] = predict1.reshape((len(predict1),1))
y_pred[jet_ind >=2] = predict2.reshape((len(predict2),1))
name = "test_result.csv"
create_csv_submission(ids, y_pred, name)
print("Prediction results are stored")