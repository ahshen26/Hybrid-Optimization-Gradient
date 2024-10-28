import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, shgo, dual_annealing
from functools import partial
import Cost_Sequential

# Parameters
N = 200
r1 = 0.5  
q1 = 1    
q2 = 1    
effectSize = 0.3962035
bias = 0
sigma = 1
alpha = 0.05
alpha_EQ = 0.2
calibration = 1

def extract_cost(params, N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration):
    split_ratio, randomization_ratio, eq_margin = params
    input_array = np.array([split_ratio, randomization_ratio, eq_margin]).reshape((3,1,1))
    _, _, treatment_arm, cost = Cost_Sequential.fun_Power(input_array, N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    return cost

# initial guesses for parameters: [split ratio, randomization ratio, equivalence margin]
x0 = np.array([0.1, 0.2, 0.4])
bounds = [(0.1, 0.9), (0.1, 0.9), (0.1, 0.5)]

cost_function = partial(extract_cost, N=N, r1=r1, q1=q1, q2=q2, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=calibration)

result_min = minimize(cost_function, x0, method='trust-constr', bounds=bounds)
result_shgo = shgo(cost_function, bounds=bounds)
result_DA = dual_annealing(cost_function, bounds=bounds)

# added treatment arm results
results = {
    'Split Ratio': [result_min.x[0], result_shgo.x[0], result_DA.x[0]],
    'Randomization Ratio': [result_min.x[1], result_shgo.x[1], result_DA.x[1]],
    'Equivalence Margin': [result_min.x[2], result_shgo.x[2], result_DA.x[2]],
    'Cost': [result_min.fun, result_shgo.fun, result_DA.fun],
    'Treatment Arm Minimize': Cost_Sequential.fun_Power(result_min.x, N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)[2][0][0],
    'Treatment Arm SHGO': Cost_Sequential.fun_Power(result_shgo.x, N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)[2][0][0],
    'Treatment Arm DA': Cost_Sequential.fun_Power(result_DA.x, N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)[2][0][0],
}

df_results = pd.DataFrame(results, index=['minimize', 'shgo', 'dual_annealing'])
df_results.to_csv(f'/Users/arlinashen/Downloads/Gradient_Optimization/Sequential_result_C{calibration}.csv', index=True)
print(f"Results saved for calibration = {calibration}")


