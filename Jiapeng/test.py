import plotly.graph_objects as go
import plotly.io as pio
import Cost_Hybrid
import numpy as np
from functools import partial
from scipy.optimize import Bounds, minimize, SR1, shgo, dual_annealing
import autograd.numpy as np
from autograd import jacobian, hessian

N = 200
q = 1
effectSize = 0.3962035 # 2.801582 * np.sqrt(sqrt(1/100 + 1/100))
bias = 0
sigma = 1
alpha = 0.05
alpha_EQ = 0.2
calibration = 4

x = np.linspace(0.1, 0.9, 20)
y = np.linspace(0.1, 0.5, 20)
x, y = np.meshgrid(x, y)
res = Cost_Hybrid.fun_Power(np.array([x,y]), N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
max_typeIerror = np.max(res[0], axis = 2)
min_power = np.min(res[1], axis = 2)
cost = res[3]

# Create a Plotly 3D surface plot
fig = go.Figure(data=[go.Surface(z=cost, x=x, y=y)])

# Update the layout for better visualization
fig.update_layout(
    title='3D Surface Plot of f(x, y) = x^2 + y^2',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='f(x, y)',
    )
)

# Show the plot in a browser
pio.renderers.default = 'browser'
fig.show()

# ##############################
Cost_Hybrid.fun_Power(np.array([0.6473684, 0.1]), N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
Cost_Hybrid.fun_Power(np.array([0.6473684, 0.28]), N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)

bounds = Bounds([0.1, 0.1], [0.9, 0.5])
def extract_cost(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration):
    # Call the original objective function
    _, _, _, cost = Cost_Hybrid.fun_Power(input, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    return cost

# Define a function to compute the numerical gradient


x0 = np.array([0.2, 0.4])

cost_function = partial(extract_cost, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=4)

result = minimize(
    fun=cost_function,
    x0=x0,
    method='trust-constr',
    jac="3-point",
    options={'verbose': 3, 'maxiter': 1000},
    bounds=bounds,
    tol = 1e-50
)
print(result.x)
print(result.fun)

bounds = [(0.1, 0.9), (0.1, 0.5)]
result_shgo = shgo(cost_function, bounds)
result_DA = dual_annealing(cost_function, bounds)
cost_function(np.array([0.55, 0.2868]))