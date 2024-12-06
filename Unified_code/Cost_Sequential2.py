import numpy as np
from scipy.integrate import quad
from probFinder import func_getProb
from scipy.stats import norm
from scipy.optimize import root_scalar

def fun_Power(N, EQ_margin, n_stage, s, r, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration):

    x1_mean_array = np.zeros(n_stage)
    x2_mean_array = np.zeros(n_stage)
    x3_mean_array = np.zeros(n_stage)
    W_mean_array = np.zeros(n_stage)
    x1_var_array = np.zeros(n_stage)
    x2_var_array = np.zeros(n_stage)
    x3_var_array = np.zeros(n_stage)
    W_var_array = np.zeros(n_stage)
    E1_array = np.zeros(n_stage)

    for i in range(n_stage):
        if i == 0 :
            w = q[i] / (1 + q[i] - r[i])
            x1_var = sigma ** 2 / (N * r[i] * (1 - r[i]))
            x2_var = sigma ** 2 * (1 + q[i] - r[i]) / (N * q[i] * (1 - r[i]))
            x3_var = sigma ** 2 * (1 + q[i]) / (N * r * (1 + q[i] - r[i]))
            x1_mean = effectSize
            x2_mean = bias
            x3_mean = effectSize - w * bias
            cov_x1x2 = sigma ** 2 / (N * (1 - r[i]))
            corr_x1x2 = cov_x1x2 / np.sqrt(x1_var) / np.sqrt(x2_var)
            theta = EQ_margin[i] - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var)

            # def integrand1(z):
            #     return z * norm.pdf(z, x2_mean, np.sqrt(x2_var))
            # def integrand2(z):
            #     return z**2 * norm.pdf(z, x2_mean, np.sqrt(x2_var))
            #
            # E1 = norm.cdf(theta, x2_mean, np.sqrt(x2_var)) - norm.cdf(-theta, x2_mean, np.sqrt(x2_var))
            # E2 = quad(integrand1, -theta, theta)[0]
            # E3 = quad(integrand2, -np.inf, -theta)[0] + quad(integrand2, theta, np.inf)[0]
            #
            # x3I_var = (x3_var + x3_mean**2)*E1 - x3_mean**2 * E1**2
            # x1I_var = E3*x1_var*corr_x1x2**2/x2_var + 2*(x1_mean*cov_x1x2/x2_var - x2_mean*x1_var*corr_x1x2**2/x2_var)*(x2_mean - E2) + (x2_mean**2*x1_var*corr_x1x2**2/x2_var + x1_mean**2 - 2*x1_mean*x2_mean*cov_x1x2/x2_var + x1_var*(1-corr_x1x2**2))*(1-E1) - ((x1_mean - x2_mean*cov_x1x2/x2_var)*(1-E1) + (x2_mean - E2)*cov_x1x2/x2_var)**2
            # cov_x1Ix3I = -x3_mean*E1*((x1_mean - x2_mean*cov_x1x2/x2_var)*(1-E1) + (x2_mean - E2)*cov_x1x2/x2_var)
            # # Need recalculate W_mean = E1*x3_mean + (1-E1)*x1_mean
            # W_var = x3I_var + x1I_var + 2*cov_x1Ix3I

            E1_array[i] = E1
        else:
            P = E1_array[i - 1]
            x1_var_1 = sigma ** 2 / (N * 0.5 * 0.5)
            x1_var_2 = sigma ** 2 / (N * r[i] * (1 - r[i]))
            x2_var_1 = sigma ** 2 * (1 + q[i] - 0.5) / (N * q[i] * (1 - 0.5))
            x2_var_2 = sigma ** 2 * (1 + q[i] - r[i]) / (N * q[i] * (1 - r[i]))
            w_1 = q[i] / (1 + q[i] - 0.5)
            w_2 = q[i] / (1 + q[i] - r[i])
            x3_var_1 = sigma ** 2 * (1 + q[i]) / (N * r * (1 + q[i] - 0.5))
            x3_var_2 = sigma ** 2 * (1 + q[i]) / (N * r * (1 + q[i] - r[i]))
            x3_mean_1 = effectSize - w_1 * bias
            x3_mean_2 = effectSize - w_2 * bias


            cov_x1x2_1 = sigma ** 2 / (N * (1 - 0.5))
            cov_x1x2_2 = sigma ** 2 / (N * (1 - r[i]))
            corr_x1x2_1 = cov_x1x2_1 / np.sqrt(x1_var_1) / np.sqrt(x2_var_1)
            corr_x1x2_2 = cov_x1x2_2 / np.sqrt(x1_var_2) / np.sqrt(x2_var_2)
            theta_1 = EQ_margin[i] - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var_1)
            theta_2 = EQ_margin[i] - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var_2)

            x1_mean = effectSize
            x1_var = P * x1_var_2 + (1 - P) * x1_var_1
            x2_mean = bias
            x2_var = P * x2_var_2 + (1 - P) * x2_var_1
            x3_mean = P * x3_mean_2 + (1 - P) * x3_mean_1
            x3_var = P * x3_var_2 + (1 - P) * x3_var_1 + P * (1 - P) * (x3_mean_2 - x3_mean_1) ** 2
            cov_x1x2 = P * cov_x1x2_2 + (1 - P) * cov_x1x2_1




    N1 = np.round(N * s)
    N2 = np.round(N * (1 - s))
    N2_t = np.round(N2 * r2)
    r2 = N2_t / N2

    # Stage 1
    w_stage1 = q1 / (1 + q1 - r1)
    x2_mean = bias
    x1_var_stage1 = sigma ** 2 / (N1 * r1 * (1 - r1))
    x2_var_stage1 = sigma ** 2 * (1 + q1 - r1) / (N1 * q1 * (1 - r1))
    x3_var_stage1 = sigma ** 2 * (1 + q1) / (N1 * r1 * (1 + q1 - r1))
    cov_x1x2 = sigma ** 2 / (N1 * (1 - r1))
    theta = EQ_margin - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var_stage1)

    if theta < 0: theta = 0

    # Stage 2: Scenario 1: The null hypothesis of the equivalence test is rejected
    w_stage2 = q2 / (1 + q2 - r2)
    x3_var_stage2_scenario1 = sigma ** 2 * (1 + q2) / (N2 * r2 * (1 + q2 - r2))

    # Stage 2: Scenario 2: we fail to reject the null hypothesis of the equivalence test
    x1_var_stage2_scenario2 = sigma ** 2 / (N2 * 0.5 * (1 - 0.5))

    # Final step: pool estimators from stage 1 and stage 2 together
    ### Scenario 2: we fail to reject the null hypothesis of the equivalence test
    v2 = x1_var_stage2_scenario2 / (x1_var_stage2_scenario2 + x1_var_stage1)
    W2_var = v2 ** 2 * x1_var_stage1 + (1 - v2) ** 2 * x1_var_stage2_scenario2
    W2_mean = effectSize
    cov_w2x2 = v2 * cov_x1x2
    corr_w2x2 = cov_w2x2 / np.sqrt(W2_var) / np.sqrt(x2_var_stage1)

    ### Scenario 1: The null hypothesis of the equivalence test is rejected
    v1 = x3_var_stage2_scenario1 / (x3_var_stage2_scenario1 + x3_var_stage1)
    W1_mean = W2_mean - (w_stage1 * v1 + w_stage2 * (1 - v1)) * bias
    W1_var = v1 ** 2 * x3_var_stage1 + (1 - v1) ** 2 * x3_var_stage2_scenario1

    if calibration == 1:
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

        cutoffValue_borr = cutoffValue_no_borr = np.sqrt(W_var) * norm.ppf(1 - alpha / 2)
        N_t = N1 * 0.5 + N2 * r2 * E1 + N2 * 0.5 * (1 - E1)

    elif calibration == 2:
        def root_function(z):
            return func_getProb(z, 0, 0, 0, 1, 1, 1, corr_w2x2, theta/ np.sqrt(x2_var_stage1), "all") - alpha

        sol = root_scalar(root_function, bracket=[0, 10], method='brenth', xtol=1e-12, maxiter=10000)
        cutoffValue_borr = sol.root * np.sqrt(W1_var)
        cutoffValue_no_borr = sol.root * np.sqrt(W2_var)

        E1 = norm.cdf(theta, x2_mean, np.sqrt(x2_var_stage1)) - norm.cdf(-theta, x2_mean, np.sqrt(x2_var_stage1))
        N_t = N1 * 0.5 + N2 * r2 * E1 + N2 * 0.5 * (1 - E1)

    elif calibration == 3:

        p = 0.54
        max_no_borr = func_getProb(0, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "outside")
        max_borr = func_getProb(0, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "inside")

        typeIerror_no_borr = alpha * p
        typeIerror_borr = alpha * (1-p)

        if typeIerror_no_borr > max_no_borr:
            cutoffValue_no_borr = 0
            def root_function_borr(z):
                return func_getProb(z, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "inside") - (alpha - max_no_borr)
            sol_borr = root_scalar(root_function_borr, bracket=[0, 10], method='brenth', xtol=1e-12, maxiter=10000)
            cutoffValue_borr = sol_borr.root
        elif typeIerror_borr > max_borr:
            cutoffValue_borr = 0
            def root_function_no_borr(z):
                return func_getProb(z, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "outside") - (alpha - max_borr)
            sol_no_borr = root_scalar(root_function_no_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
            cutoffValue_no_borr = sol_no_borr.root
        else:
            def root_function_no_borr(z):
                return func_getProb(z, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "outside") - typeIerror_no_borr
            sol_no_borr = root_scalar(root_function_no_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
            cutoffValue_no_borr = sol_no_borr.root
            def root_function_borr(z):
                return func_getProb(z, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "inside") - typeIerror_borr
            sol_borr = root_scalar(root_function_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
            cutoffValue_borr = sol_borr.root

        E1 = norm.cdf(theta, x2_mean, np.sqrt(x2_var_stage1)) - norm.cdf(-theta, x2_mean, np.sqrt(x2_var_stage1))
        N_t = N1 * 0.5 + N2 * r2 * E1 + N2 * 0.5 * (1 - E1)

    else:
        cutoffValue_no_borr = norm.ppf(1 - alpha/2)*np.sqrt(W2_var)

        typeIerror_no_borr4 = func_getProb(cutoffValue_no_borr, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "outside")
        if typeIerror_no_borr4 >= alpha:
            cutoffValue_borr = np.inf
        else:
            def root_function_borr(z):
                return func_getProb(z, 0, 0, 0, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "inside") - (alpha - typeIerror_no_borr4)
            sol_borr = root_scalar(root_function_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
            cutoffValue_borr = sol_borr.root
        E1 = norm.cdf(theta, x2_mean, np.sqrt(x2_var_stage1)) - norm.cdf(-theta, x2_mean, np.sqrt(x2_var_stage1))
        N_t = N1 * 0.5 + N2 * r2 * E1 + N2 * 0.5 * (1 - E1)
    typeIerror = np.array([])
    power = np.array([])

    for j in np.arange(-0.6, 0.61, 0.05):
        if theta == 0:
            typeIerror = np.append(typeIerror, norm.cdf(-cutoffValue_no_borr, 0, np.sqrt(W2_var)) + (1 - norm.cdf(cutoffValue_no_borr, 0, np.sqrt(W2_var))))
            power = np.append(power, norm.cdf(-cutoffValue_no_borr, W2_mean, np.sqrt(W2_var)) + (1 - norm.cdf(cutoffValue_no_borr, W2_mean, np.sqrt(W2_var))))
        else:
            typeIerror = np.append(typeIerror, func_getProb(cutoffValue_no_borr, 0, j, - (w_stage1 * v1 + w_stage2 * (1 - v1)) * j, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "outside") + func_getProb(cutoffValue_borr, 0, j, -v1*w_stage1*j - (1-v1)*w_stage2*j, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "inside"))
            W1_mean = W2_mean - (w_stage1 * v1 + w_stage2 * (1 - v1)) * j
            power = np.append(power, func_getProb(cutoffValue_no_borr, W2_mean, j, W1_mean, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "outside") + func_getProb(cutoffValue_borr, W2_mean, j, W1_mean, W2_var, x2_var_stage1, W1_var, cov_w2x2, theta, "inside"))

    # dimension = len(input.shape)
    #
    # if dimension == 3:
    #     n_col = input.shape[1]
    #     n_row = input.shape[2]
    #     typeIerror_array = np.zeros((n_col, n_row, len(np.arange(-0.6, 0.61, 0.05))))
    #     power_array = np.zeros((n_col, n_row, len(np.arange(-0.6, 0.61, 0.05))))
    #     nTreatmentArm_array = np.zeros((n_col, n_row))
    #     cost_array = np.zeros((n_col, n_row))
    # elif dimension == 2:
    #     n_col = input.shape[0]
    #     n_row = 1
    #     typeIerror_array = np.zeros((n_col, len(np.arange(-0.6, 0.61, 0.05))))
    #     power_array = np.zeros((n_col, len(np.arange(-0.6, 0.61, 0.05))))
    #     nTreatmentArm_array = np.zeros((n_col))
    #     cost_array = np.zeros((n_col))
    # else:
    #     n_col = 1
    #     n_row = 1
    #     typeIerror_array = np.zeros((len(np.arange(-0.6, 0.61, 0.05))))
    #     power_array = np.zeros((len(np.arange(-0.6, 0.61, 0.05))))
    #     nTreatmentArm_array = 0
    #     cost_array = 0
    #
    # for m in range(n_col):
    #     for n in range(n_row):
    #         if dimension == 3:
    #             s = input[0, m, n]
    #             r2 = input[1, m, n]
    #             EQ_margin = input[2, m, n]  # 0.3
    #         elif dimension == 2:
    #             s = input[m, 0]
    #             r2 = input[m, 1]
    #             EQ_margin = input[m, 2]  # 0.3
    #         else:
    #             s = input[0]
    #             r2 = input[1]
    #             EQ_margin = input[2]  # 0.3
    #
    #
    #         if dimension == 3:
    #             typeIerror_array[m, n, :] = typeIerror
    #             power_array[m, n, :] = power
    #             nTreatmentArm_array[m, n] = N_t
    #             cost_array[m, n] = (1 - power[len(power) // 2] - 0.005 * N_t + (100*(max(typeIerror)-0.07) if max(typeIerror) > 0.07 else 0) + (100*(0.77 - min(power)) if min(power) < 0.77 else 0))*100
    #
    #         elif dimension == 2:
    #             typeIerror_array[m, :] = typeIerror
    #             power_array[m, :] = power
    #             nTreatmentArm_array[m] = N_t
    #             cost_array[m] = (1 - power[len(power) // 2] - 0.005 * N_t + (100*(max(typeIerror)-0.07) if max(typeIerror) > 0.07 else 0) + (100*(0.77 - min(power)) if min(power) < 0.77 else 0))*100
    #
    #         else:
    #             typeIerror_array = typeIerror
    #             power_array = power
    #             nTreatmentArm_array = N_t
    #             cost_array = (1 - power[len(power) // 2] - 0.005 * N_t + (100*(max(typeIerror)-0.07) if max(typeIerror) > 0.07 else 0) + (100*(0.77 - min(power)) if min(power) < 0.77 else 0))*100

    return typeIerror_array, power_array, nTreatmentArm_array, cost_array