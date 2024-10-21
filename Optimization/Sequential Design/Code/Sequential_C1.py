import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm, multivariate_normal

def fun_Power1(input):
    s = input[0]
    r2 = input[1]
    EQ_margin = input[2]

    N = 200
    r1 = 0.5
    q1 = 1
    q2 = 1
    effectSize = 2.801582
    bias = 0
    sigma = 1
    alpha = 0.05
    alpha_EQ = 0.2

    N1 = int(N * s)
    N2 = int(N * (1 - s))
    N2_t = int(N2 * r2)
    r2 = N2_t / N2

    # Stage 1
    w_stage1 = q1 / (1 + q1 - r1)
    x2_mean = bias
    x1_var_stage1 = sigma ** 2 / (N1 * r1 * (1 - r1))
    x2_var_stage1 = sigma ** 2 * (1 + q1 - r1) / (N1 * q1 * (1 - r1))
    x3_var_stage1 = sigma ** 2 * (1 + q1) / (N1 * r1 * (1 + q1 - r1))
    cov_x1x2 = sigma ** 2 / (N1 * (1 - r1))
    theta = EQ_margin - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var_stage1)

    if theta <= 0: return 0, 0, np.inf

    # Stage 2: Scenario 1: The null hypothesis of the equivalence test is rejected
    w_stage2 = q2 / (1 + q2 - r2)
    x3_var_stage2_scenario1 = sigma ** 2 * (1 + q2) / (N2 * r2 * (1 + q2 - r2))

    # Stage 2: Scenario 2: we fail to reject the null hypothesis of the equivalence test
    x1_var_stage2_scenario2 = sigma ** 2 / (N2 * 0.5 * (1 - 0.5))

    # Final step: pool estimators from stage 1 and stage 2 together
    ### Scenario 2: we fail to reject the null hypothesis of the equivalence test
    v2 = x1_var_stage2_scenario2 / (x1_var_stage2_scenario2 + x1_var_stage1)
    W2_var = v2 ** 2 * x1_var_stage1 + (1 - v2) ** 2 * x1_var_stage2_scenario2
    W2_mean = effectSize * np.sqrt(W2_var)
    cov_w2x2 = v2 * cov_x1x2
    corr_w2x2 = cov_w2x2 / np.sqrt(W2_var) / np.sqrt(x2_var_stage1)

    ### Scenario 1: The null hypothesis of the equivalence test is rejected
    v1 = x3_var_stage2_scenario1 / (x3_var_stage2_scenario1 + x3_var_stage1)
    W1_mean = effectSize * np.sqrt(W2_var) - (w_stage1 * v1 + w_stage2 * (1 - v1)) * bias
    W1_var = v1 ** 2 * x3_var_stage1 + (1 - v1) ** 2 * x3_var_stage2_scenario1

    def integrand1(z):
        return z * norm.pdf(z, x2_mean, np.sqrt(x2_var_stage1))
    def integrand2(z):
        return z**2 * norm.pdf(z, x2_mean, np.sqrt(x2_var_stage1))

    E1 = norm.cdf(theta, x2_mean, np.sqrt(x2_var_stage1)) - norm.cdf(-theta, x2_mean, np.sqrt(x2_var_stage1))
    E2 = quad(integrand1, -theta, theta)[0]
    E3 = quad(integrand2, -np.inf, -theta)[0] + quad(integrand2, theta, np.inf)[0]

    W1I_var = (W1_var + W1_mean**2)*E1 - W1_mean**2 * E1**2
    W2I_var = E3*W2_var*corr_w2x2**2/x2_var_stage1 + 2*(W2_mean*cov_w2x2/x2_var_stage1 - x2_mean*W2_var*corr_w2x2**2/x2_var_stage1)*(x2_mean - E2) + (x2_mean**2*W2_var*corr_w2x2**2/x2_var_stage1 + W2_mean**2 - 2*W2_mean*x2_mean*cov_w2x2/x2_var_stage1 + W2_var*(1-corr_w2x2**2))*(1-E1) - ((W2_mean - x2_mean*cov_w2x2/x2_var_stage1)*(1-E1) + (x2_mean - E2)*cov_w2x2/x2_var_stage1)**2
    cov_W1IW2I = -W1_mean*E1*((W2_mean - x2_mean*cov_w2x2/x2_var_stage1)*(1-E1) + (x2_mean - E2)*cov_w2x2/x2_var_stage1)
    W_var = W1I_var + W2I_var + 2*cov_W1IW2I

    # ### Bivariate normal distribution
    # def integrand1_middle(x, y):
    #     return x*multivariate_normal.pdf(np.array([x, y]), mean=np.array([W1_mean, x2_mean]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
    # def integrand2_middle(x, y):
    #     return x**2 * multivariate_normal.pdf(np.array([x, y]), mean=np.array([W1_mean, x2_mean]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
    # def integrand1_outside(x, y):
    #     return x*multivariate_normal.pdf(np.array([x, y]), mean=np.array([W2_mean, x2_mean]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
    # def integrand2_outside(x, y):
    #     return x**2 *multivariate_normal.pdf(np.array([x, y]), mean=np.array([W2_mean, x2_mean]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
    #
    # W1I_var2 = dblquad(integrand2_middle,  -theta, theta, lambda y: -np.inf, lambda y: np.inf)[0]  - (dblquad(integrand1_middle,  -theta, theta, lambda y: -np.inf, lambda y: np.inf)[0])**2
    # W2I_var2 = dblquad(integrand2_outside, -np.inf, -theta, lambda y: -np.inf, lambda y: np.inf)[0] + dblquad(integrand2_outside, theta, np.inf, lambda y: -np.inf, lambda y: np.inf)[0] - (dblquad(integrand1_outside, -np.inf, -theta, lambda y: -np.inf, lambda y: np.inf)[0] + dblquad(integrand1_outside, theta, np.inf, lambda y: -np.inf, lambda y: np.inf)[0]) ** 2
    # cov_W1IW2I2 = -(dblquad(integrand1_middle,  -theta, theta, lambda y: -np.inf, lambda y: np.inf)[0])*((dblquad(integrand1_outside, -np.inf, -theta, lambda y: -np.inf, lambda y: np.inf)[0]) + (dblquad(integrand1_outside, theta, np.inf, lambda y: -np.inf, lambda y: np.inf)[0]))
    # W_var = W1I_var2 + W2I_var2 + 2*cov_W1IW2I2


    cutoffValue = np.sqrt(W_var) * norm.ppf(1 - alpha / 2) # This cutoffValue is unstandardized

    typeIerror_list = []
    power_list = []
    for j in np.arange(-0.6, 0.61, 0.05):
        # Non-borrow: upperleft
        typeIerror_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Non-borrow: upperright
        typeIerror_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Non-borrow: buttomright
        typeIerror_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Non-borrow: buttomleft
        typeIerror_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Borrow: upper
        typeIerror_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
        # Borrow: buttom
        typeIerror_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))

        typeIerror = typeIerror_ul + typeIerror_ur + typeIerror_br + typeIerror_bl + typeIerror_u + typeIerror_b
        typeIerror_list.append(typeIerror)

        W1_mean = effectSize * np.sqrt(W2_var) - (w_stage1 * v1 + w_stage2 * (1 - v1)) * j
        # Non-borrow: upperleft
        Power_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Non-borrow: upperright
        Power_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Non-borrow: buttomright
        Power_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Non-borrow: buttomleft
        Power_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # Borrow: upper
        Power_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
        # Borrow: buttom
        Power_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))

        Power = Power_ul + Power_ur + Power_br + Power_bl + Power_u + Power_b
        power_list.append(Power)

    Cost = 1 - power_list[len(power_list)//2] + (np.inf if max(typeIerror_list) > 0.07 else 0) + (np.inf if min(power_list) < 0.77 else 0)

    return typeIerror_list, power_list, Cost

def fun_Cost1(input):

    n_particles = input.shape[0]
    Cost = []

    for i in range(n_particles):
        s = input[i][0]
        r2 = input[i][1]
        EQ_margin = input[i][2]

        N = 200
        r1 = 0.5
        q1 = 1
        q2 = 1
        effectSize = 2.801582
        bias = 0
        sigma = 1
        alpha = 0.05
        alpha_EQ = 0.2

        N1 = int(N * s)
        N2 = int(N * (1 - s))
        N2_t = int(N2 * r2)
        r2 = N2_t / N2

        # Stage 1
        w_stage1 = q1 / (1 + q1 - r1)
        x2_mean = bias
        x1_var_stage1 = sigma ** 2 / (N1 * r1 * (1 - r1))
        x2_var_stage1 = sigma ** 2 * (1 + q1 - r1) / (N1 * q1 * (1 - r1))
        x3_var_stage1 = sigma ** 2 * (1 + q1) / (N1 * r1 * (1 + q1 - r1))
        cov_x1x2 = sigma ** 2 / (N1 * (1 - r1))
        theta = EQ_margin - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var_stage1)

        if theta <= 0:
            Cost.append(np.inf)
            continue

        # Stage 2: Scenario 1: The null hypothesis of the equivalence test is rejected
        w_stage2 = q2 / (1 + q2 - r2)
        x3_var_stage2_scenario1 = sigma ** 2 * (1 + q2) / (N2 * r2 * (1 + q2 - r2))

        # Stage 2: Scenario 2: we fail to reject the null hypothesis of the equivalence test
        x1_var_stage2_scenario2 = sigma ** 2 / (N2 * 0.5 * (1 - 0.5))

        # Final step: pool estimators from stage 1 and stage 2 together
        ### Scenario 2: we fail to reject the null hypothesis of the equivalence test
        v2 = x1_var_stage2_scenario2 / (x1_var_stage2_scenario2 + x1_var_stage1)
        W2_var = v2 ** 2 * x1_var_stage1 + (1 - v2) ** 2 * x1_var_stage2_scenario2
        W2_mean = effectSize * np.sqrt(W2_var)
        cov_w2x2 = v2 * cov_x1x2
        corr_w2x2 = cov_w2x2 / np.sqrt(W2_var) / np.sqrt(x2_var_stage1)

        ### Scenario 1: The null hypothesis of the equivalence test is rejected
        v1 = x3_var_stage2_scenario1 / (x3_var_stage2_scenario1 + x3_var_stage1)
        W1_mean = effectSize * np.sqrt(W2_var) - (w_stage1 * v1 + w_stage2 * (1 - v1)) * bias
        W1_var = v1 ** 2 * x3_var_stage1 + (1 - v1) ** 2 * x3_var_stage2_scenario1

        def integrand1(z):
            return z * norm.pdf(z, x2_mean, np.sqrt(x2_var_stage1))
        def integrand2(z):
            return z**2 * norm.pdf(z, x2_mean, np.sqrt(x2_var_stage1))

        E1 = norm.cdf(theta, x2_mean, np.sqrt(x2_var_stage1)) - norm.cdf(-theta, x2_mean, np.sqrt(x2_var_stage1))
        E2 = quad(integrand1, -theta, theta)[0]
        E3 = quad(integrand2, -np.inf, -theta)[0] + quad(integrand2, theta, np.inf)[0]

        W1I_var = (W1_var + W1_mean**2)*E1 - W1_mean**2 * E1**2
        W2I_var = E3*W2_var*corr_w2x2**2/x2_var_stage1 + 2*(W2_mean*cov_w2x2/x2_var_stage1 - x2_mean*W2_var*corr_w2x2**2/x2_var_stage1)*(x2_mean - E2) + (x2_mean**2*W2_var*corr_w2x2**2/x2_var_stage1 + W2_mean**2 - 2*W2_mean*x2_mean*cov_w2x2/x2_var_stage1 + W2_var*(1-corr_w2x2**2))*(1-E1) - ((W2_mean - x2_mean*cov_w2x2/x2_var_stage1)*(1-E1) + (x2_mean - E2)*cov_w2x2/x2_var_stage1)**2
        cov_W1IW2I = -W1_mean*E1*((W2_mean - x2_mean*cov_w2x2/x2_var_stage1)*(1-E1) + (x2_mean - E2)*cov_w2x2/x2_var_stage1)
        W_var = W1I_var + W2I_var + 2*cov_W1IW2I

        # ### Bivariate normal distribution
        # def integrand1_middle(x, y):
        #     return x*multivariate_normal.pdf(np.array([x, y]), mean=np.array([W1_mean, x2_mean]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
        # def integrand2_middle(x, y):
        #     return x**2 * multivariate_normal.pdf(np.array([x, y]), mean=np.array([W1_mean, x2_mean]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
        # def integrand1_outside(x, y):
        #     return x*multivariate_normal.pdf(np.array([x, y]), mean=np.array([W2_mean, x2_mean]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        # def integrand2_outside(x, y):
        #     return x**2 *multivariate_normal.pdf(np.array([x, y]), mean=np.array([W2_mean, x2_mean]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
        #
        # W1I_var2 = dblquad(integrand2_middle,  -theta, theta, lambda y: -np.inf, lambda y: np.inf)[0]  - (dblquad(integrand1_middle,  -theta, theta, lambda y: -np.inf, lambda y: np.inf)[0])**2
        # W2I_var2 = dblquad(integrand2_outside, -np.inf, -theta, lambda y: -np.inf, lambda y: np.inf)[0] + dblquad(integrand2_outside, theta, np.inf, lambda y: -np.inf, lambda y: np.inf)[0] - (dblquad(integrand1_outside, -np.inf, -theta, lambda y: -np.inf, lambda y: np.inf)[0] + dblquad(integrand1_outside, theta, np.inf, lambda y: -np.inf, lambda y: np.inf)[0]) ** 2
        # cov_W1IW2I2 = -(dblquad(integrand1_middle,  -theta, theta, lambda y: -np.inf, lambda y: np.inf)[0])*((dblquad(integrand1_outside, -np.inf, -theta, lambda y: -np.inf, lambda y: np.inf)[0]) + (dblquad(integrand1_outside, theta, np.inf, lambda y: -np.inf, lambda y: np.inf)[0]))
        # W_var = W1I_var2 + W2I_var2 + 2*cov_W1IW2I2


        cutoffValue = np.sqrt(W_var) * norm.ppf(1 - alpha / 2) # This cutoffValue is unstandardized

        typeIerror_list = []
        power_list = []
        for j in np.arange(-0.6, 0.61, 0.05):
            # Non-borrow: upperleft
            typeIerror_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Non-borrow: upperright
            typeIerror_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Non-borrow: buttomright
            typeIerror_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Non-borrow: buttomleft
            typeIerror_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([0, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Borrow: upper
            typeIerror_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
            # Borrow: buttom
            typeIerror_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([-v1*w_stage1*j - (1-v1)*w_stage2*j, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))

            typeIerror = typeIerror_ul + typeIerror_ur + typeIerror_br + typeIerror_bl + typeIerror_u + typeIerror_b
            typeIerror_list.append(typeIerror)

            W1_mean = effectSize * np.sqrt(W2_var) - (w_stage1 * v1 + w_stage2 * (1 - v1)) * j
            # Non-borrow: upperleft
            Power_ul = multivariate_normal.cdf([np.inf, -theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Non-borrow: upperright
            Power_ur = 1 - multivariate_normal.cdf([np.inf, theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, np.inf], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Non-borrow: buttomright
            Power_br = multivariate_normal.cdf([-cutoffValue, np.inf], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Non-borrow: buttomleft
            Power_bl = multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([W2_mean, j]), cov=np.array([[W2_var, cov_w2x2], [cov_w2x2, x2_var_stage1]]))
            # Borrow: upper
            Power_u = multivariate_normal.cdf([np.inf, theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([cutoffValue, theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([np.inf, -theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) + multivariate_normal.cdf([cutoffValue, -theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))
            # Borrow: buttom
            Power_b = multivariate_normal.cdf([-cutoffValue, theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]])) - multivariate_normal.cdf([-cutoffValue, -theta], mean=np.array([W1_mean, j]), cov=np.array([[W1_var, 0], [0, x2_var_stage1]]))

            Power = Power_ul + Power_ur + Power_br + Power_bl + Power_u + Power_b
            power_list.append(Power)

        Cost.append(1 - power_list[len(power_list)//2] + (np.inf if max(typeIerror_list) > 0.07 else 0) + (np.inf if min(power_list) < 0.77 else 0))

    return np.array(Cost)