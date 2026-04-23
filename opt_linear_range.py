import numpy as np
import scipy.stats as stats

def linear_range_obj_fn(X,Y,l,slope_ub=np.inf,slope_lb=-np.inf):
    """Objective function for finding the optimal linear range

    Args:
        X (array-like): independent variable
        Y (array-like): dependent variable
        l (float): regularization parameter
    """

    # estimate linear regression

    res = stats.linregress(X,Y)

    # if slope_ub is not None:
    if res.slope > slope_ub:
        return np.inf
    
    if res.slope < slope_lb:
        return np.inf
    
    else:
        n = len(X)

        return (1-res.rvalue**2) + l/(n**2)
        # return (1-res.rvalue) + l/n

def opt_linear_range(X,Y,l,slope_ub=np.inf,slope_lb=-np.inf):
    """Finds the optimal linear range for a given dataset

    Args:
        X (array-like): independent variable
        Y (array-like): dependent variable
        l (float): regularization parameter
    """

    if type(X) is list:
        X = np.array(X)
    if type(Y) is list:
        Y = np.array(Y)
    
    # for each subset of X, compute the objective function
    n = len(X)

    loss_list = []
    subset_list = []

    # print('\n')
    # print('n = ' + str(n))
    
    for subset_length in range(2,n):
        for start in range(n-subset_length):
            end = start + subset_length
            X_subset = X[start:end]
            Y_subset = Y[start:end]

            # print(end)

            loss = linear_range_obj_fn(X_subset,Y_subset,l,slope_ub=slope_ub,slope_lb=slope_lb)
            loss_list.append(loss)
            subset_list.append((start,end))
    
    # find the subset with the minimum loss
    min_loss_indx = np.argmin(loss_list)

    return subset_list[min_loss_indx]