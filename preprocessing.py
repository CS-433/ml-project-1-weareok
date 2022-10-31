#%%%%%%%%%%%%%%%%%%% Higgs Boson Machine Learning%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#    preprocessing.py includes data preprocessing code for Higgs Boson Machine Learning project, it has three modules: 
#    1) data partition module (based on indicator Jet_num), 2) data filtering (to mask the meaningless sample and reject outlier data), 
#    3) polynomial building module(to allow the model with higher degrees of dependency on the feature)
#    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from featurefiltering import *
from featureprocessing import *



# degree choice for ridge regression
degree_model0 =  9
degree_model1 =  10
degree_model2 =  10

# degree choice for logistic regression
#degree_model0 =  4
#degree_model1 =  4
#degree_model2 =  4

def  data_partition(y_label, x_feature):
    """
    Partition the data set according to the jet number
    
    Args:
        y_label: labels for training
        x_feature: feature vectors for regression 
    
    Returns:
        y_labeli: i = 0,...,3
        x_featurei: i = 0,...,3
    """
    
    # get the sample index for three subgroups 
    Index_jetnum0 =  x_feature[:,22] == 0
    Index_jetnum1 =  x_feature[:,22] == 1
    Index_jetnum2 =  x_feature[:,22] >= 2
    
    # partition the original label into four subgroups 
    y_label0 = y_label[Index_jetnum0]
    y_label1 = y_label[Index_jetnum1]
    y_label2 = y_label[Index_jetnum2]

    
    # partition the original feature into four subgroups 
    
    x_feature0  = x_feature[Index_jetnum0]
    x_feature1  = x_feature[Index_jetnum1] 
    x_feature2  = x_feature[Index_jetnum2]

    
    return y_label0,y_label1,y_label2,x_feature0,x_feature1,x_feature2
    
def preprocessing(y_label, x_feature):
    # partition training data
    y_label0, y_label1, y_label2, x_feature0, x_feature1, x_feature2 = data_partition(y_label, x_feature)

    # delete the undefined feature for the data 
    x_feature0 = feature_delete(x_feature0, indicator = 0)
    x_feature1 = feature_delete(x_feature1, indicator = 1)
    x_feature2 = feature_delete(x_feature2, indicator = 2)

    # dealing with missing data
    x_feature0 = missing_to_zero(x_feature0)
    x_feature1 = missing_to_zero(x_feature1)
    x_feature2 = missing_to_zero(x_feature2)

    # dealing with outliners 
    x_feature0 =  outlier_to_zero(x_feature0,  threshold  = 0.996)
    x_feature1 =  outlier_to_zero(x_feature1,  threshold  = 0.989)
    x_feature2 =  outlier_to_zero(x_feature2,  threshold  = 0.975)

    # replace outliers by the feature mean
    x_feature0 = replace_zeros_by_mean(x_feature0)
    x_feature1 = replace_zeros_by_mean(x_feature1)
    x_feature2 = replace_zeros_by_mean(x_feature2)

    # standardize feature data 
    x_std0, mean_x0, std_x0 = standardize(x_feature0)
    x_std1, mean_x1, std_x1 = standardize(x_feature1)
    x_std2, mean_x2, std_x2 = standardize(x_feature2)
    
    print("Starting to build polynomial expression for the feature matrix")
    # build polynomial model
    x_output0 =  build_polynomial(x_std0, degree_model0, group = 1)
    x_output1 =  build_polynomial(x_std1, degree_model1, group = 2)
    x_output2 =  build_polynomial(x_std2, degree_model2, group = 3)
    #x_output0 =  build_polynomial(x_std0, 9)
    #x_output1 =  build_polynomial(x_std1, 10)
    #x_output2 =  build_polynomial(x_std2, 10)
    print("Polynomial forms are built")
    
    return [y_label0, y_label1, y_label2], [x_output0, x_output1, x_output2]

