import pandas as pd
import Sequential_C2 as C2
from mealpy.swarm_based.CSO import OriginalCSO
from mealpy import FloatVar
import numpy as np
import pickle

# Set-up Bounds

# Split ratio between stage 1 and stage 2
# Randomization ratio;
# Equivalence margin;
# Split proportion of type I error rate
min_bound = np.array([0.1, 0.1, 0])
max_bound = np.array([0.9, 0.9, 1])

#############################################################################################
######################################### CSO ###############################################
#############################################################################################

# Calibration 2
def extract_cost2(input):
    # Call the original objective function
    _, _, cost = C2.fun_Power2(input)
    return cost

problem2 = {
    "obj_func": extract_cost2,    # Objective function
    "bounds": FloatVar(lb=min_bound, ub=max_bound),
    "minmax": "min",                   # Minimize the objective function
    "verbose": True                    # Print the progress
}


model2 = OriginalCSO(
    epoch=200,        # Number of iterations
    pop_size=100,      # Population size
)

Results2 = model2.solve(problem=problem2, n_workers=8)

with open('log/sequential_history_CSO_C2.pkl', 'wb') as f:
    pickle.dump(model2.history, f)

result2_CSO = C2.fun_Power2(Results2.solution)

combined_data = [int(200*Results2.solution[0])/200, int(200*Results2.solution[0]*Results2.solution[1])/int(200*Results2.solution[0]), Results2.solution[2]] + result2_CSO[1] + result2_CSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result2_CSO = pd.DataFrame([combined_data])
DF_result2_CSO.columns = ['Split ratio between stage 1 and stage 2', 'Randomization Ratio', 'Equivalence Margin'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result2_CSO.to_csv('Results/Sequential_result2_CSO.csv', index=False)