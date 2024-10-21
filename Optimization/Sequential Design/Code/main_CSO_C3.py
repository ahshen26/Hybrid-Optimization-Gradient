import pandas as pd
import Sequential_C3 as C3
from mealpy.swarm_based.CSO import OriginalCSO
from mealpy import FloatVar
import numpy as np
import pickle

# Set-up Bounds

# Split ratio between stage 1 and stage 2
# Randomization ratio;
# Equivalence margin;
# Split proportion of type I error rate
min_bound = np.array([0.1, 0.1, 0, 0.1])
max_bound = np.array([0.9, 0.9, 1, 0.9])

#############################################################################################
######################################### CSO ###############################################
#############################################################################################

# Calibration 3
def extract_cost3(input):
    # Call the original objective function
    _, _, cost = C3.fun_Power3(input)
    return cost

problem3 = {
    "obj_func": extract_cost3,    # Objective function
    "bounds": FloatVar(lb=min_bound, ub=max_bound),
    "minmax": "min",                   # Minimize the objective function
    "verbose": True                    # Print the progress
}


model3 = OriginalCSO(
    epoch=200,        # Number of iterations
    pop_size=100,      # Population size
)

Results3 = model3.solve(problem=problem3, n_workers=8)

with open('log/sequential_history_CSO_C3.pkl', 'wb') as f:
    pickle.dump(model3.history, f)

result3_CSO = C3.fun_Power3(Results3.solution)

combined_data = [int(200*Results3.solution[0])/200, int(200*Results3.solution[0]*Results3.solution[1])/int(200*Results3.solution[0]), Results3.solution[2], Results3.solution[3]] + result3_CSO[1] + result3_CSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result3_CSO = pd.DataFrame([combined_data])
DF_result3_CSO.columns = ['Split ratio between stage 1 and stage 2', 'Randomization Ratio', 'Equivalence Margin', 'Split proportion of type I error rate'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result3_CSO.to_csv('Results/Sequential_result3_CSO.csv', index=False)