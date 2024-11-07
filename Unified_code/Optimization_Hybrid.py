import numpy as np
import pandas as pd
import os
import Cost_Hybrid
import pyswarms as ps
import CSOMA_Python
from mealpy.swarm_based.CSO import OriginalCSO
from mealpy import FloatVar
from scipy.optimize import dual_annealing
from functools import partial

# Set-up Bounds

# Randomization ratio;
# Equivalence margin;
N = 200
q = 1
effectSize = 0.3962035
bias = 0
sigma = 1
alpha = 0.05
alpha_EQ = 0.2

def extract_cost(input, weight, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration):
    # Call the original objective function
    _, _, _, _, cost = Cost_Hybrid.fun_Power(input, weight, N, q, effectSize, bias, sigma, alpha, alpha_EQ, calibration)
    return cost

DF = pd.DataFrame()
for weight in np.array([0,0.005]):
    for i in range(1,5):
        if i == 3:
            dimensions = 3
            min_bound = np.array([0.1, 0.1, 0.1])
            max_bound = np.array([0.9, 0.5, 0.9])
            bounds = [(0.1, 0.9), (0.1, 0.5), (0.1, 0.9)]
            x0 = np.array([0.2, 0.4, 0.1])
        else:
            dimensions = 2
            min_bound = np.array([0.1, 0.1])
            max_bound = np.array([0.9, 0.5])
            bounds = [(0.1, 0.9), (0.1, 0.5)]
            x0 = np.array([0.2, 0.4])
        n_particles = 50
        iters = 200
        cost_function = partial(extract_cost, weight=weight, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=i)
        
        #############################################################################################
        ######################################### PSO ###############################################
        #############################################################################################
        # Set-up Hyperparameters
        options_PSO = {'c1': 0.5, 'c2': 0.3, 'w': 0.7}

        # Call instance of PSO
        optimizer_PSO = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options_PSO, bounds=(min_bound, max_bound))
        # Perform optimization
        cost_PSO, pos_PSO = optimizer_PSO.optimize(cost_function, iters=iters)
        result_PSO = Cost_Hybrid.fun_Power(pos_PSO, weight = weight, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=i)
        #############################################################################################
        ######################################### CSOMA #############################################
        #############################################################################################
        # Set-up Hyperparameters
        options_CSOMA = {'c1': 0.5, 'c2': 0.3, 'w': 0.7, 'phi': 0.2}

        # Call instance of PSO
        optimizer_CSOMA = CSOMA_Python.single.CSOMA(n_particles=n_particles, dimensions=dimensions, options=options_CSOMA,bounds=(min_bound, max_bound))
        # Perform optimization
        cost_CSOMA, pos_CSOMA = optimizer_CSOMA.optimize(cost_function, iters=iters)
        result_CSOMA = Cost_Hybrid.fun_Power(pos_CSOMA, weight = weight, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=i)
        #############################################################################################
        ######################################### CSO ###############################################
        #############################################################################################
        problem = {
        "obj_func": cost_function,    # Objective function
        "bounds": FloatVar(lb=min_bound, ub=max_bound),
        "minmax": "min",                   # Minimize the objective function
        "verbose": True                    # Print the progress
        }

        model = OriginalCSO(
            epoch=iters,        # Number of iterations
            pop_size=n_particles,      # Population size
        )

        fittedModel = model.solve(problem=problem, n_workers=8)
        result_CSO = Cost_Hybrid.fun_Power(fittedModel.solution, weight = weight, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=i)
        #############################################################################################
        ######################################### DUAL ANNEALING #############################################
        #############################################################################################
        fitted_results = dual_annealing(cost_function, bounds)
        result_DA = Cost_Hybrid.fun_Power(fitted_results.x, weight = weight, N=N, q=q, effectSize=effectSize, bias=bias, sigma=sigma, alpha=alpha, alpha_EQ=alpha_EQ, calibration=i)

        if i == 3:
            combined_data_PSO = [weight, i, round(200*pos_PSO[0], 0)/200, round(200*pos_PSO[0], 0), pos_PSO[1], pos_PSO[2], cost_PSO, result_PSO[2], result_PSO[3], result_PSO[4]] + list(result_PSO[1]) + list(result_PSO[0])
            combined_data_CSO = [weight, i, round(200*fittedModel.solution[0], 0)/200, round(200*fittedModel.solution[0], 0), fittedModel.solution[1], fittedModel.solution[2], fittedModel.target.objectives[0], result_CSO[2], result_CSO[3], result_CSO[4]] + list(result_CSO[1]) + list(result_CSO[0])
            combined_data_CSOMA = [weight, i, round(200*pos_CSOMA[0], 0)/200, round(200*pos_CSOMA[0], 0), pos_CSOMA[1], pos_CSOMA[2], np.float64(cost_CSOMA), result_CSOMA[2], result_CSOMA[3], result_CSOMA[4]] + list(result_CSOMA[1]) + list(result_CSOMA[0])
            combined_data_DA = [weight, i, round(200*fitted_results.x[0], 0)/200, round(200*fitted_results.x[0], 0), fitted_results.x[1], fitted_results.x[2], fitted_results.fun, result_DA[2], result_DA[3], result_DA[4]] + list(result_DA[1]) + list(result_DA[0])

            all_combined_data = [combined_data_PSO, combined_data_CSO, combined_data_CSOMA, combined_data_DA]
            all_combined_data = [[round(num, 4) for num in sublist] for sublist in all_combined_data]
            columns = ["Weight", "Calibration", 'Randomization Ratio', 'N_t', 'Equivalence Margin', "Split Ratio", "Cost", "N_t", "Beta", "Cost"] + \
                      [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + \
                      [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]
            DF_result = pd.DataFrame(all_combined_data, columns=columns)
        else:
            combined_data_PSO = [weight, i, round(200*pos_PSO[0], 0)/200, round(200*pos_PSO[0], 0), pos_PSO[1], 0, cost_PSO, result_PSO[2], result_PSO[3], result_PSO[4]] + list(result_PSO[1]) + list(result_PSO[0])
            combined_data_CSO = [weight, i, round(200*fittedModel.solution[0], 0)/200, round(200*fittedModel.solution[0], 0), fittedModel.solution[1], 0, fittedModel.target.objectives[0], result_CSO[2], result_CSO[3], result_CSO[4]] + list(result_CSO[1]) + list(result_CSO[0])
            combined_data_CSOMA = [weight, i, round(200*pos_CSOMA[0], 0)/200, round(200*pos_CSOMA[0], 0), pos_CSOMA[1], 0, np.float64(cost_CSOMA), result_CSOMA[2], result_CSOMA[3], result_CSOMA[4]] + list(result_CSOMA[1]) + list(result_CSOMA[0])
            combined_data_DA = [weight, i, round(200*fitted_results.x[0], 0)/200, round(200*fitted_results.x[0], 0), fitted_results.x[1], 0, fitted_results.fun, result_DA[2], result_DA[3], result_DA[4]] + list(result_DA[1]) + list(result_DA[0])

            all_combined_data = [combined_data_PSO, combined_data_CSO, combined_data_CSOMA, combined_data_DA]
            all_combined_data = [[round(num, 4) for num in sublist] for sublist in all_combined_data]
            columns = ["Weight", "Calibration", 'Randomization Ratio', 'N_t', 'Equivalence Margin', "Split Ratio", "Cost", "N_t", "Beta", "Cost"] + \
                      [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + \
                      [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]
            DF_result = pd.DataFrame(all_combined_data, columns=columns)
        DF = pd.concat([DF, DF_result], ignore_index=True)
DF.to_csv(os.path.join(os.getcwd(), 'Optimal_design.csv'), index=False)
