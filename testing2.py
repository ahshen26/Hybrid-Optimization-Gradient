import Cost_Hybrid
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import Bounds, minimize, shgo, dual_annealing

# Define parameters
N = 200
q = 1
effectSize = 0.3962035
bias = 0
sigma = 1
alpha = 0.05
alpha_EQ = 0.2
x0 = np.array([0.2, 0.4, 0.5])  # Initial values for all three parameters
bounds = [(0.1, 0.9), (0.1, 0.5), (0.01, 0.99)]  # Bounds for each parameter

def extract_cost(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration = 3):
    if calibration == 3: 
        input = np.append(input, [0.5])  # adjust value as needed


    _, _, _, cost = Cost_Hybrid.fun_Power(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration = 3)
    return cost

for calibration in range(3, 5):
    print(f"\nRunning optimization for calibration = {calibration}")
    cost_function = partial(extract_cost, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration = calibration)

    result_min = minimize(
        fun=cost_function,
        x0=x0,
        method='trust-constr',
        jac="3-point",
        options={'verbose': 3, 'maxiter': 1000},
        bounds=Bounds([0.1, 0.1, 0.01], [0.9, 0.5, 0.99])
    )
    result_min_power = Cost_Hybrid.fun_Power(result_min.x, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)

    result_shgo = shgo(cost_function, bounds=bounds)
    result_shgo_power = Cost_Hybrid.fun_Power(result_shgo.x, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)

    result_DA = dual_annealing(cost_function, bounds=bounds)
    result_DA_power = Cost_Hybrid.fun_Power(result_DA.x, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)

    # results
    results = {
        'Randomization Ratio': [result_min.x[0], result_shgo.x[0], result_DA.x[0]],
        'Equivalence Margin': [result_min.x[1], result_shgo.x[1], result_DA.x[1]],
        'Cost': [result_min.fun, result_shgo.fun, result_DA.fun],
        **{f'Power_{round(i, 4)}': [result_min_power[1][idx], result_shgo_power[1][idx], result_DA_power[1][idx]] for idx, i in enumerate(np.arange(-0.6, 0.61, 0.05))},
        **{f'TypeIError_{round(i, 4)}': [result_min_power[0][idx], result_shgo_power[0][idx], result_DA_power[0][idx]] for idx, i in enumerate(np.arange(-0.6, 0.61, 0.05))}
    }
    DF_results = pd.DataFrame(results, index=['minimize', 'shgo', 'dual_annealing'])

    # Save each result to csv
    DF_results.to_csv(f'/Users/arlinashen/Downloads/Gradient_Optimization/Sequential_result_C{calibration}.csv', index=True)
    print(f"Results saved for calibration = {calibration}")

print("Optimization completed for all calibrations.")


