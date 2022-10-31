import numpy as np


# Index of the undefined features to be deleted (according to the documentation: The Higgs Boson machine learning challenge, Appendix B) 
jet_num_delete_0 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
jet_num_delete_1 = [4, 5, 6, 12, 22, 26, 27, 28]
jet_num_delete_23 = [22]

def feature_delete(x_feature,indicator):
    
    """
    Delete the undefined feature columns according to the jet number
    
    Args:
        x_feature: feature matrix
        indicator: jet number indicator 
    
    Returns:
        A new feature matrix without undefined data set
    """
    if indicator == 0:
        return np.delete(x_feature, jet_num_delete_0, axis = 1)
    elif indicator == 1:
        return np.delete(x_feature, jet_num_delete_1, axis = 1)
    else:
        return np.delete(x_feature, jet_num_delete_23, axis = 1)

    
def missing_to_zero(x_feature):
    """
    Ensuring the contribution of missing value for computing mean is zero,       
    otherwise mean value could be baised by the meaningless entries
    
    Args:
        x_feature: feature matrix
    
    Returns:
        feature matrix with missing entries replaced by zeros 
    """
    
    
    indx = x_feature == -999
    x_feature[indx] = 0
    return x_feature
    

def outlier_to_zero(x_feature,threshold):
    """
    Ensuring the contribution of outliers for computing mean is zero,       
    otherwise mean value could be baised by the outliers
    
    Args:
        x_feature: feature matrix
        threshold: to decide the range of the outliers to be rejected
    Returns:
        feature matrix with outliers replaced by zeros 
    """
    
    #  unreasonable threshold, return x_feature
    if threshold > 1 or threshold <0:
        return x_feature
    
    #  outliers is justified by their significance, 
    #  entriers smaller than (1- threshold) quantile and larger than threshold quantiles are 
    #  considered to be outliers
    for idx in range(x_feature.shape[1]):
        th_upper = np.quantile(x_feature[:,idx], threshold)
        th_lower = np.quantile(x_feature[:,idx], 1- threshold)
        index_outliner = np.logical_or(x_feature[:,idx]>th_upper, x_feature[:,idx]<th_lower)
        x_feature[index_outliner,idx] = 0
        
    return  x_feature

    

def replace_zeros_by_mean(x_feature):
    """
    Ensuring the contribution of outliers for computing mean is zero,       
    otherwise mean value could be baised by the outliers
    
    Args:
        x_feature: feature matrix
        threshold: to decide the range of the outliers to be rejected
    Returns:
        feature matrix with outliers replaced by zeros 
    """
    # compute mean value for each feature vector
    mean_vec = np.mean(x_feature,axis = 0)
    
    # replace missing values and outliers by the mean value 
    # such that their contribution to the modeling is moderate
    
    for idx in range(x_feature.shape[1]):       
        zero_idx = x_feature[:,idx] == 0
        x_feature[zero_idx,idx] = mean_vec[idx]
        
    return x_feature