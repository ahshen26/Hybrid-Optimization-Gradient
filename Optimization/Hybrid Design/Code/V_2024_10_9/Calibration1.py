import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, multivariate_normal

def fun_Power1(input):

    r = input[0]
    EQ_margin = input[1] # 0.3

    N = 200
    q = 1
    effectSize = 2.801582
    bias = 0
    sigma = 1
    alpha = 0.05
    alpha_EQ = 0.2
    N_t = int(N * r)
    r = N_t / N

    w = q / (1 + q - r)
    x1_var = sigma**2 / (N * r * (1 - r))
    x2_var = sigma**2 * (1 + q - r) / (N * q * (1 - r))
    x3_var = sigma**2 * (1 + q) / (N * r * (1 + q - r))
    x1_mean = effectSize * np.sqrt(x1_var)
    x2_mean = bias
    x3_mean = x1_mean - w * x2_mean
    theta = EQ_margin - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var)

    if theta <= 0: return 0, 0, np.inf

    cov_Z1Z2 = sigma**2 / (N * (1 - r))

    def integrand1(z):
        return z * norm.pdf(z, x2_mean, np.sqrt(x2_var))

    def integrand2(z):
        return z**2 * norm.pdf(z, x2_mean, np.sqrt(x2_var))

    E1, _ = quad(integrand1, -theta, theta)
    E2, _ = quad(integrand2, -theta, theta)

    W_var = x1_var + w**2 * (E2 - E1**2) - 2 * w * (E2 - x2_mean * E1) * cov_Z1Z2 / x2_var

    cutoffValue = np.sqrt(W_var) * norm.ppf(1 - alpha / 2)

    typeIerror_list = []
    power_list = []
    for j in np.arange(-0.6, 0.61, 0.05):
        # Non-borrow: upperleft
        typeIerror_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Non-borrow: upperright
        typeIerror_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Non-borrow: buttomright
        typeIerror_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Non-borrow: buttomleft
        typeIerror_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Borrow: upper
        typeIerror_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))
        # Borrow: buttom
        typeIerror_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))

        typeIerror = typeIerror_ul + typeIerror_ur + typeIerror_br + typeIerror_bl + typeIerror_u + typeIerror_b
        typeIerror_list.append(typeIerror)

        x3_mean = x1_mean - w * j
        # Non-borrow: upperleft
        Power_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Non-borrow: upperright
        Power_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Non-borrow: buttomright
        Power_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Non-borrow: buttomleft
        Power_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
        # Borrow: upper
        Power_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))
        # Borrow: buttom
        Power_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))

        Power = Power_ul + Power_ur + Power_br + Power_bl + Power_u + Power_b
        power_list.append(Power)

    Cost = 1 - power_list[len(power_list)//2] + (np.inf if max(typeIerror_list) > 0.07 else 0) + (100 if min(power_list) < 0.77 else 0)

    return typeIerror_list, power_list, Cost

def fun_Cost1(input):

    n_particles = input.shape[0]
    Cost = []

    for i in range(n_particles):

        r = input[i][0]
        EQ_margin = input[i][1]  # 0.3

        N = 200
        q = 1
        effectSize = 2.801582
        bias = 0
        sigma = 1
        alpha = 0.05
        alpha_EQ = 0.2
        N_t = int(N * r)
        r = N_t / N

        w = q / (1 + q - r)
        x1_var = sigma ** 2 / (N * r * (1 - r))
        x2_var = sigma ** 2 * (1 + q - r) / (N * q * (1 - r))
        x3_var = sigma ** 2 * (1 + q) / (N * r * (1 + q - r))
        x1_mean = effectSize * np.sqrt(x1_var)
        x2_mean = bias
        x3_mean = x1_mean - w * x2_mean
        theta = EQ_margin - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var)

        if theta <= 0:
            Cost.append(np.inf)
            continue

        cov_Z1Z2 = sigma**2 / (N * (1 - r))

        def integrand1(z):
            return z * norm.pdf(z, x2_mean, np.sqrt(x2_var))

        def integrand2(z):
            return z**2 * norm.pdf(z, x2_mean, np.sqrt(x2_var))

        E1, _ = quad(integrand1, -theta, theta)
        E2, _ = quad(integrand2, -theta, theta)

        W_var = x1_var + w**2 * (E2 - E1**2) - 2 * w * (E2 - x2_mean * E1) * cov_Z1Z2 / x2_var

        cutoffValue = np.sqrt(W_var) * norm.ppf(1 - alpha / 2)

        typeIerror_list = []
        power_list = []
        for j in np.arange(-0.6, 0.61, 0.05):
            # Non-borrow: upperleft
            typeIerror_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Non-borrow: upperright
            typeIerror_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Non-borrow: buttomright
            typeIerror_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Non-borrow: buttomleft
            typeIerror_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Borrow: upper
            typeIerror_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))
            # Borrow: buttom
            typeIerror_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([-w*j, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))

            typeIerror = typeIerror_ul + typeIerror_ur + typeIerror_br + typeIerror_bl + typeIerror_u + typeIerror_b
            typeIerror_list.append(typeIerror)

            x3_mean = x1_mean - w * j
            # Non-borrow: upperleft
            Power_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Non-borrow: upperright
            Power_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Non-borrow: buttomright
            Power_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Non-borrow: buttomleft
            Power_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([x1_mean, j]), cov=np.array([[x1_var, cov_Z1Z2], [cov_Z1Z2, x2_var]]))
            # Borrow: upper
            Power_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))
            # Borrow: buttom
            Power_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([x3_mean, j]), cov=np.array([[x3_var, 0], [0, x2_var]]))

            Power = Power_ul + Power_ur + Power_br + Power_bl + Power_u + Power_b
            power_list.append(Power)

        Cost.append(1 - power_list[len(power_list)//2] + (np.inf if max(typeIerror_list) > 0.07 else 0) + (100 if min(power_list) < 0.77 else 0))

    return np.array(Cost)