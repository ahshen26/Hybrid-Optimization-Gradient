import numpy as np
from scipy.integrate import quad
from probFinder import func_getProb
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize


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
                cost_array[m, n] = (1 - power[len(power) // 2] - 0.5 * N_t + (1000 * max(typeIerror) - 0.07) * (1000 * (max(typeIerror) - 0.07) if max(typeIerror) > 0.07 else 0) + (1000 * 0.77 - min(power)) * (1000 * (0.77 - min(power)) if min(power) < 0.77 else 0))
            elif dimension == 2:
                typeIerror_array[m, :] = typeIerror
                power_array[m, :] = power
                nTreatmentArm_array[m] = N_t
                cost_array[m] = (
                1 - power[len(power) // 2]
                - 0.5 * N_t
                + (1000 * max(typeIerror) - 0.07) * (1000 * (max(typeIerror) - 0.07) if max(typeIerror) > 0.07 else 0)
                + (1000 * 0.77 - min(power)) * (1000 * (0.77 - min(power)) if min(power) < 0.77 else 0))

            else:
                typeIerror_array = typeIerror
                power_array = power
                nTreatmentArm_array = N_t
                cost_array = 1 - power[len(power) // 2] - 0.5 * N_t + (1000*max(typeIerror) - 0.07)(1000*(max(typeIerror)-0.07) if max(typeIerror) > 0.07 else 0) + (1000*0.77 - min(power))(1000*(0.77 - min(power)) if min(power) < 0.77 else 0)

    return typeIerror_array, power_array, nTreatmentArm_array, cost_array
# def objective_function(params, input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration):
#     r, EQ_margin = params  
# # It is (2,) or (2,x) or (2,x,x)
# #The first one is the first parameter, the second one is the second parameter 
# #Each parameter can be a single number a 1d array, or a 2d array
#     # adjust input based on its shape
#     modified_input = input.copy()

#     if modified_input.ndim == 3:
#         modified_input[0, :, :] = r  # update r for 3d
#         modified_input[1, :, :] = EQ_margin  # update EQ for 3d input
#     elif modified_input.ndim == 2:
#         modified_input[0, :] = r  # update r for 2d
#         modified_input[1, :] = EQ_margin  # update EQ for 2d input
#     elif modified_input.ndim == 1:
#         modified_input[0] = r  # update r for 1d
#         modified_input[1] = EQ_margin  # update EQ_margin for 1d input

#     # get the cost_array 
#     _, _, _, cost_array = fun_Power(modified_input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)

#     # im returning the mean cost but can use a diffferent metric for optimizing 
#     return np.mean(cost_array)

# def optimize_parameters(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration, initial_r, initial_EQ_margin):
#     initial_params = [initial_r, initial_EQ_margin]
#     bounds = [(0.1, 0.9), (0, 1)]  # bounds 

#     result = minimize(objective_function, initial_params, args=(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration), bounds=bounds, method='Nelder-Mead')
    
#     if result.success:
#         optimized_r, optimized_EQ_margin = result.x
#         print(f"Optimized r: {optimized_r}, Optimized EQ_margin: {optimized_EQ_margin}")
#         return optimized_r, optimized_EQ_margin
#     else:
#         print("Optimization failed.")
#         print(result)  
#         return None, None

# # example
# N = 200  
# q = 1.0  
# effectSize = 0.5  #
# bias = 0.1  #
# sigma = 1.0  
# alpha = 0.05  
# alpha_EQ = 0.05  
# calibration = 1  #
# initial_r = 0.5  #  guess for r
# initial_EQ_margin = 0.25  # guess for EQ_margin


# a 2D array input with size (2, 10)
#input_data = np.random.rand(2, 10)  # replace with actual input data

#optimized_r, optimized_EQ_margin = optimize_parameters(input_data, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration, initial_r, initial_EQ_margin)

# NEWTON RAPHSON METHOD

def newton_raphson(params, input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration, epsilon=1e-6):
    max_iterations = 100
    r, EQ_margin = params

    # define cost function
    def cost_function(params):
        modified_input = input.copy()
        if modified_input.ndim == 3:
            modified_input[0, :, :] = params[0]
            modified_input[1, :, :] = params[1]
        elif modified_input.ndim == 2:
            modified_input[0, :] = params[0]
            modified_input[1, :] = params[1]
        else:
            modified_input[0] = params[0]
            modified_input[1] = params[1]

        _, _, _, cost_array = fun_Power(modified_input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
        return np.mean(cost_array)


    for _ in range(max_iterations):
        f_value = cost_function(params)
        
        # numerical approximation of the gradient using finite differences
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_temp = params.copy()
            params_temp[i] += epsilon
            gradient[i] = (cost_function(params_temp) - f_value) / epsilon

        # update parameters using Newton-Raphson method
        params -= f_value / gradient

    return params
# sample data
N = 200  
q = 1.0  
effectSize = 0.5  
bias = 0.1  
sigma = 1.0  
alpha = 0.05  
alpha_EQ = 0.05  
calibration = 1  
initial_r = 0.5  
initial_EQ_margin = 0.25  

# dummy input data
input_data = np.random.rand(2, 10)  # 2 rows for r and EQ_margin

# run newton method
optimized_params = newton_raphson(
    np.array([initial_r, initial_EQ_margin]),
    input_data,
    N,
    q,
    effectSize,
    bias,
    sigma,
    alpha,
    alpha_EQ,
    calibration
)

print("Optimized parameters:", optimized_params)



