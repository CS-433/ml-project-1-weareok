#%%%%%%%%%%%%%%%%%%% Higgs Boson Machine Learning%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#    featureprocessing.py includes some tools to mapping the label (for logistic regression)
#    standardize the data to avoid the inbalance between different features as well as to build higher
#    degree of dependencies between different features
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import numpy as np





def logistic_label_mapping(y_label):
    """
    Ensuring the contribution of outliers for computing mean is zero,       
    otherwise mean value could be baised by the outliers
    
    Args:
        y_label: label vector
    Returns:
        labels with -1 ----> 0 
    """
    idx = y_label == -1
    y_label[idx] = 0
    return y_label

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis = 0)
    print(np.shape(mean_x))
    x = x - mean_x
    std_x = np.std(x,axis = 0)
    x = x / std_x
    return x, mean_x, std_x


def build_polynomial(x_std,degree, group):
    
    data_number     =  x_std.shape[0]   #  number of data sample  
    N  =  x_std.shape[1]   #  dimension of feature vector
    x_output = np.ones((data_number,1))
    
    # fecture vector with a specified degrees 
    # [1, x_1, x_1^2,...,x_1^N,....,1,x_D, x_D^2, ..., x_D^N]
    for ite in range(degree):
        x_power = x_std**ite
        x_output = np.concatenate((x_output, x_power),axis = 1)
    
    # add the cross-term (of first order) into the regression vector
    # [x_1x_2, ..., x_1x_D, ...,x_{D-1}x_{D}]

    for out_loop in range(N - 1):
        vec1 = np.expand_dims(x_std[:, out_loop],axis = 1)
        x_new        =  vec1*(x_std[:, out_loop+1:-1])   
        x_output =  np.concatenate((x_output, x_new),axis = 1)


    vec2 = x_std**2
    # add the cross-term(with degree 2) into the regression vector
    # [x_1^2x_2, ..., x_1^2x_D, ...,x_{D-1}^2x_{D}]
    for out_loop in range(N - 1):
        vec1 = np.expand_dims(x_std[:, out_loop],axis = 1)
        x_new1       =  vec1*vec2[:, out_loop+1:-1]
        x_output =  np.concatenate((x_output, x_new1),axis = 1)

    # add the cross-term(with degree 2) into the regression vector
    # [x_1x_2^2, ..., x_1x_D^2, ...,x_{D-1}x_{D}^2]
    for out_loop in range(N - 1):
        vec1 = np.expand_dims(vec2[:, out_loop],axis = 1)
        x_new2        =  vec1*(x_std[:, out_loop+1:-1])
        x_output =  np.concatenate((x_output, x_new2),axis = 1)


    # add the cross-term(with degree 2) into the regression vector
    # [x_1^2x_2^2, ..., x_1^2x_D^2, ...,x_{D-1}^2x_{D}^2]
    for out_loop in range(N - 1):
        vec   = np.expand_dims(vec2[:, out_loop], axis = 1)
        x_new3 = vec * (vec2[:, out_loop + 1:-1])
        x_output = np.concatenate((x_output, x_new3), axis=1)




    print("Polynomial feature has been built")

    return  x_output