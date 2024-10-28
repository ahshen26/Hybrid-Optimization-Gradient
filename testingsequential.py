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
calibration = 4


def extract_cost(input, N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration):
    _, _, _, cost = Cost_Sequential.fun_Power(np.array(input).reshape((3,1,1)), N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    return cost 


x0 = np.array([0.2, 0.4])
bounds = [(0.1, 0.9), (0.1, 0.5)]

cost_function = partial(extract_cost, N=N, r1=r1, q1=q1, q2=q2, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=calibration)


result_min = minimize(cost_function, x0, method='trust-constr', bounds=bounds)
result_shgo = shgo(cost_function, bounds=bounds)
result_DA = dual_annealing(cost_function, bounds=bounds)

intervals = np.arange(-0.6, 0.61, 0.05)
power_data = {}

for val in intervals:
    _, powers_min, _, _ = Cost_Sequential.fun_Power(np.array([result_min.x[0], result_min.x[1], val]).reshape((3,1,1)), N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    _, powers_shgo, _, _ = Cost_Sequential.fun_Power(np.array([result_shgo.x[0], result_shgo.x[1], val]).reshape((3,1,1)), N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    _, powers_DA, _, _ = Cost_Sequential.fun_Power(np.array([result_DA.x[0], result_DA.x[1], val]).reshape((3,1,1)), N, r1, q1, q2, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    power_data[f'Power_{round(val, 4)}'] = [powers_min[0][0], powers_shgo[0][0], powers_DA[0][0]]  # Extract the first power result

results = {
    'Randomization Ratio': [result_min.x[0], result_shgo.x[0], result_DA.x[0]],
    'Equivalence Margin': [result_min.x[1], result_shgo.x[1], result_DA.x[1]],
    'Cost': [result_min.fun, result_shgo.fun, result_DA.fun],
    **power_data
}

df_results = pd.DataFrame(results, index=['minimize', 'shgo', 'dual_annealing'])
df_results.to_csv(f'/Users/arlinashen/Downloads/Gradient_Optimization/Sequential_result_C{calibration}.csv', index=True)
print(f"Results saved for calibration = {calibration}")
