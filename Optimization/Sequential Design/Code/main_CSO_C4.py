import pandas as pd
import Sequential_C4 as C4
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

# Calibration 4
def extract_cost4(input):
    # Call the original objective function
    _, _, cost = C4.fun_Power4(input)
    return cost

problem4 = {
    "obj_func": extract_cost4,    # Objective function
    "bounds": FloatVar(lb=min_bound, ub=max_bound),
    "minmax": "min",                   # Minimize the objective function
    "verbose": True                    # Print the progress
}


model4 = OriginalCSO(
    epoch=200,        # Number of iterations
    pop_size=100,      # Population size
)

Results4 = model4.solve(problem=problem4, n_workers=8)

with open('log/sequential_history_CSO_C4.pkl', 'wb') as f:
    pickle.dump(model4.history, f)

result4_CSO = C4.fun_Power4(Results4.solution)

combined_data = [int(200*Results4.solution[0])/200, int(200*Results4.solution[0]*Results4.solution[1])/int(200*Results4.solution[0]), Results4.solution[2]] + result4_CSO[1] + result4_CSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result4_CSO = pd.DataFrame([combined_data])
DF_result4_CSO.columns = ['Split ratio between stage 1 and stage 2', 'Randomization Ratio', 'Equivalence Margin'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result4_CSO.to_csv('Results/Sequential_result4_CSO.csv', index=False)