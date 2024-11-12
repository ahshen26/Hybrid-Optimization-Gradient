import numpy as np
from scipy.stats import multivariate_normal

def func_getProb(cutoffValue, x1_mean, x2_mean, x3_mean, x1_var, x2_var, x3_var, cov, theta, area):

    # Non-borrow: upperleft
    p_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]]))
    # Non-borrow: upperright
    p_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]]))
    # Non-borrow: buttomright
    p_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]]))
    # Non-borrow: buttomleft
    p_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([x1_mean, x2_mean]), cov=np.array([[x1_var, cov], [cov, x2_var]]))

    # Borrow: upper
    p_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([x3_mean, x2_mean]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([x3_mean, x2_mean]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([x3_mean, x2_mean]), cov=np.array([[x3_var, 0], [0, x2_var]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([x3_mean, x2_mean]), cov=np.array([[x3_var, 0], [0, x2_var]]))
    # Borrow: buttom
    p_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([x3_mean, x2_mean]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([x3_mean, x2_mean]), cov=np.array([[x3_var, 0], [0, x2_var]]))

    if area == "outside":
        p = p_ul + p_ur + p_br + p_bl
        return p

    elif area == "inside":
        p = p_u + p_b
        return p

    elif area == "all":
        p = p_ul + p_ur + p_br + p_bl + p_u + p_b
        return p