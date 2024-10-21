import numpy as np
from scipy.stats import multivariate_normal

def fun_cutoffValueFinder(cutoffValue, x1_mean, x2_mean, x3_mean, x1_var, x2_var, x3_var, corr, theta):

    # Non-borrow: upperleft
    p_ul = multivariate_normal.cdf([np.inf, -theta/np.sqrt(x2_var)], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]])) - multivariate_normal.cdf([cutoffValue, -theta/np.sqrt(x2_var)], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]])) 
    # Non-borrow: upperright
    p_ur = 1 - multivariate_normal.cdf([np.inf, theta/np.sqrt(x2_var)], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]])) + multivariate_normal.cdf([cutoffValue, theta/np.sqrt(x2_var)], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]])) 
    # Non-borrow: buttomright
    p_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]])) - multivariate_normal.cdf([-cutoffValue, theta/np.sqrt(x2_var)], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]]))
    # Non-borrow: buttomleft
    p_bl = multivariate_normal.cdf([-cutoffValue, -theta/np.sqrt(x2_var)], mean=np.array([x1_mean/np.sqrt(x1_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, corr], [corr, 1]])) 
    # Borrow: upper
    p_u = multivariate_normal.cdf([np.inf, theta/np.sqrt(x2_var)], mean=np.array([x3_mean/np.sqrt(x3_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, 0], [0, 1]])) - multivariate_normal.cdf([cutoffValue, theta/np.sqrt(x2_var)], mean=np.array([x3_mean/np.sqrt(x3_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, 0], [0, 1]])) - multivariate_normal.cdf([np.inf, -theta/np.sqrt(x2_var)], mean=np.array([x3_mean/np.sqrt(x3_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, 0], [0, 1]])) + multivariate_normal.cdf([cutoffValue, -theta/np.sqrt(x2_var)], mean=np.array([x3_mean/np.sqrt(x3_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, 0], [0, 1]])) 
    # Borrow: buttom
    p_b = multivariate_normal.cdf([-cutoffValue, theta/np.sqrt(x2_var)], mean=np.array([x3_mean/np.sqrt(x3_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, 0], [0, 1]])) - multivariate_normal.cdf([-cutoffValue, -theta/np.sqrt(x2_var)], mean=np.array([x3_mean/np.sqrt(x3_var), x2_mean/np.sqrt(x2_var)]), cov=np.array([[1, 0], [0, 1]])) 

    p1 = p_ul + p_ur + p_br + p_bl 
    p2 = p_u + p_b

    p = p1 + p2

    return p, p1, p2