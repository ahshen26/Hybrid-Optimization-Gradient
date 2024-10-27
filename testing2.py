import plotly.graph_objects as go
import plotly.io as pio
import Cost_Hybrid
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import Bounds, minimize, shgo, dual_annealing

# Define parameters
N = 200
q = 1
effectSize = 0.3962035  # 2.801582 * np.sqrt(sqrt(1/100 + 1/100))
bias = 0
sigma = 1
alpha = 0.05
alpha_EQ = 0.2
calibration = 2

x = np.linspace(0.1, 0.9, 20)
y = np.linspace(0.1, 0.5, 20)
x, y = np.meshgrid(x, y)
res = Cost_Hybrid.fun_Power(np.array([x, y]), N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
cost = res[3]

fig = go.Figure(data=[go.Surface(z=cost, x=x, y=y)])
fig.update_layout(
    title='3D Surface Plot of Cost Function',
    scene=dict(xaxis_title='Randomization Ratio', yaxis_title='Equivalence Margin', zaxis_title='Cost')
)
fig.show()

# Optimization function
def extract_cost(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration):
    _, _, _, cost = Cost_Hybrid.fun_Power(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    return cost


x0 = np.array([0.2, 0.4])
bounds = [(0.1, 0.9), (0.1, 0.5)]
cost_function = partial(extract_cost, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=calibration)

# Perform optimizations
result_min = minimize(fun=cost_function, x0=x0, method='trust-constr', jac="3-point", options={'verbose': 3, 'maxiter': 1000}, bounds=Bounds([0.1, 0.1], [0.9, 0.5]))
result_shgo = shgo(cost_function, bounds=bounds)
result_DA = dual_annealing(cost_function, bounds=bounds)

result_min_power = Cost_Hybrid.fun_Power(result_min.x, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
result_shgo_power = Cost_Hybrid.fun_Power(result_shgo.x, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
result_DA_power = Cost_Hybrid.fun_Power(result_DA.x, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)

results = {
    'Randomization Ratio': [result_min.x[0], result_shgo.x[0], result_DA.x[0]],
    'Equivalence Margin': [result_min.x[1], result_shgo.x[1], result_DA.x[1]],
    'Cost': [result_min.fun, result_shgo.fun, result_DA.fun],
    **{f'Power_{round(i, 4)}': [result_min_power[1][idx], result_shgo_power[1][idx], result_DA_power[1][idx]] for idx, i in enumerate(np.arange(-0.6, 0.61, 0.05))},
    **{f'TypeIError_{round(i, 4)}': [result_min_power[0][idx], result_shgo_power[0][idx], result_DA_power[0][idx]] for idx, i in enumerate(np.arange(-0.6, 0.61, 0.05))}
}


DF_results = pd.DataFrame(results, index=['minimize', 'shgo', 'dual_annealing'])


DF_results.to_csv(f'/Users/arlinashen/Downloads/Gradient_Optimization/Sequential_result_C{calibration}.csv', index=True)
print(f"Results saved for calibration = {calibration}")


