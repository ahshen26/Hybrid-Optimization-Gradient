import numpy as np
from scipy.integrate import quad
from probFinder import func_getProb
from scipy.stats import norm
from scipy.optimize import root_scalar

def fun_Power(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration):
    dimension = len(input.shape)

    if dimension == 3:
        n_col = input.shape[1]
        n_row = input.shape[2]
        typeIerror_array = np.zeros((n_col, n_row, len(np.arange(-0.6, 0.61, 0.05))))
        power_array = np.zeros((n_col, n_row, len(np.arange(-0.6, 0.61, 0.05))))
        nTreatmentArm_array = np.zeros((n_col, n_row))
        cost_array = np.zeros((n_col, n_row))
    elif dimension == 2:
        n_col = input.shape[1]
        n_row = 1
        typeIerror_array = np.zeros((n_col, len(np.arange(-0.6, 0.61, 0.05))))
        power_array = np.zeros((n_col, len(np.arange(-0.6, 0.61, 0.05))))
        nTreatmentArm_array = np.zeros((n_col))
        cost_array = np.zeros((n_col))
    else:
        n_col = 1
        n_row = 1
        typeIerror_array = np.zeros((len(np.arange(-0.6, 0.61, 0.05))))
        power_array = np.zeros((len(np.arange(-0.6, 0.61, 0.05))))
        nTreatmentArm_array = 0
        cost_array = 0

    for m in range(n_col):
        for n in range(n_row):
            if dimension == 3:
                r = input[0, m, n]
                EQ_margin = input[1, m, n]  # 0.3
            elif dimension == 2:
                r = input[0, m]
                EQ_margin = input[1, m]  # 0.3
            else:
                r = input[0]
                EQ_margin = input[1]  # 0.3

            N_t = int(N * r)
            r = N_t / N

            w = q / (1 + q - r)
            x1_var = sigma**2 / (N * r * (1 - r))
            x2_var = sigma**2 * (1 + q - r) / (N * q * (1 - r))
            x3_var = sigma**2 * (1 + q) / (N * r * (1 + q - r))
            x1_mean = effectSize
            x2_mean = bias
            theta = EQ_margin - norm.ppf(1 - alpha_EQ / 2) * np.sqrt(x2_var)
            cov_Z1Z2 = sigma ** 2 / (N * (1 - r))

            if theta <= 0:
                if dimension == 3:
                    typeIerror_array[m, n, :] = np.zeros(len(np.arange(-0.6, 0.61, 0.05)))
                    power_array[m, n, :] = np.zeros(len(np.arange(-0.6, 0.61, 0.05)))
                    nTreatmentArm_array[m, n] = 0
                    cost_array[m, n] = -1000*theta

                elif dimension == 2:
                    typeIerror_array[m, :] = np.zeros(len(np.arange(-0.6, 0.61, 0.05)))
                    power_array[m, :] = np.zeros(len(np.arange(-0.6, 0.61, 0.05)))
                    nTreatmentArm_array[m] = 0
                    cost_array[m] = -1000*theta

                else:
                    typeIerror_array = np.zeros(len(np.arange(-0.6, 0.61, 0.05)))
                    power_array = np.zeros(len(np.arange(-0.6, 0.61, 0.05)))
                    nTreatmentArm_array = 0
                    cost_array = -1000*theta

                continue

            if calibration == 1:
                def integrand1(z):
                    return z * norm.pdf(z, x2_mean, np.sqrt(x2_var))

                def integrand2(z):
                    return z**2 * norm.pdf(z, x2_mean, np.sqrt(x2_var))

                E1, _ = quad(integrand1, -theta, theta)
                E2, _ = quad(integrand2, -theta, theta)

                W_var = x1_var + w**2 * (E2 - E1**2) - 2 * w * (E2 - x2_mean * E1) * cov_Z1Z2 / x2_var

                cutoffValue_borr = np.sqrt(W_var) * norm.ppf(1 - alpha / 2)
                cutoffValue_no_borr = cutoffValue_borr
            elif calibration == 2:
                def root_function(z):
                    return func_getProb(z, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "all") - alpha

                sol = root_scalar(root_function, bracket=[0, 10], method='brenth', xtol=1e-12, maxiter=10000)
                cutoffValue_borr = cutoffValue_no_borr = sol.root
            elif calibration == 3:

                p = input[2]

                max_no_borr = func_getProb(0, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "outside")
                max_borr = func_getProb(0, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "inside")

                typeIerror_no_borr = alpha * p
                typeIerror_borr = alpha * (1-p)

                if typeIerror_no_borr > max_no_borr:
                    cutoffValue_no_borr = 0
                    def root_function_borr(z):
                        return func_getProb(z, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "inside")[2] - (alpha - max_no_borr)
                    sol_borr = root_scalar(root_function_borr, bracket=[0, 10], method='brenth', xtol=1e-12, maxiter=10000)
                    cutoffValue_borr = sol_borr.root
                elif typeIerror_borr > max_borr:
                    cutoffValue_borr = 0
                    def root_function_no_borr(z):
                        return func_getProb(z, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "outside") - (alpha - max_borr)
                    sol_no_borr = root_scalar(root_function_no_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
                    cutoffValue_no_borr = sol_no_borr.root
                else:
                    def root_function_no_borr(z):
                        return func_getProb(z, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "outside") - typeIerror_no_borr
                    sol_no_borr = root_scalar(root_function_no_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
                    cutoffValue_no_borr = sol_no_borr.root
                    def root_function_borr(z):
                        return func_getProb(z, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "inside") - typeIerror_borr
                    sol_borr = root_scalar(root_function_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
                    cutoffValue_borr = sol_borr.root
            else:
                cutoffValue_no_borr = norm.ppf(1 - alpha/2)

                if func_getProb(cutoffValue_no_borr, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "outside") >= alpha:
                    cutoffValue_borr = np.inf
                else:
                    def root_function_borr(z):
                        return func_getProb(z, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "inside") - (alpha - func_getProb(cutoffValue_no_borr, 0, 0, 0, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "outside"))
                    sol_borr = root_scalar(root_function_borr, bracket=[0, 10], method= 'brenth', xtol=1e-12, maxiter=10000)
                    cutoffValue_borr = sol_borr.root

            typeIerror = np.array([])
            power = np.array([])
            for j in np.arange(-0.6, 0.61, 0.05):

                typeIerror = np.append(typeIerror, func_getProb(cutoffValue_no_borr, 0, j, -w*j, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "outside") + func_getProb(cutoffValue_borr, 0, j, -w*j, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "inside"))

                x3_mean = x1_mean - w * j
                power = np.append(power, func_getProb(cutoffValue_no_borr, x1_mean, j, x3_mean, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "outside") + func_getProb(cutoffValue_borr, x1_mean, j, x3_mean, x1_var, x2_var, x3_var, cov_Z1Z2, theta, "inside"))

            if dimension == 3:
                typeIerror_array[m, n, :] = typeIerror
                power_array[m, n, :] = power
                nTreatmentArm_array[m, n] = N_t
                cost_array[m, n] = 1 - power[len(power) // 2] - 0.5 * N_t + (1000*(max(typeIerror)-0.07) if max(typeIerror) > 0.07 else 0) + (1000*(0.77 - min(power)) if min(power) < 0.77 else 0)

            elif dimension == 2:
                typeIerror_array[m, :] = typeIerror
                power_array[m, :] = power
                nTreatmentArm_array[m] = N_t
                cost_array[m] = 1 - power[len(power) // 2] - 0.5 * N_t + (1000*(max(typeIerror)-0.07) if max(typeIerror) > 0.07 else 0) + (1000*(0.77 - min(power)) if min(power) < 0.77 else 0)

            else:
                typeIerror_array = typeIerror
                power_array = power
                nTreatmentArm_array = N_t
                cost_array = 1 - power[len(power) // 2] - 0.5 * N_t + (1000*(max(typeIerror)-0.07) if max(typeIerror) > 0.07 else 0) + (1000*(0.77 - min(power)) if min(power) < 0.77 else 0)

    return typeIerror_array, power_array, nTreatmentArm_array, cost_array