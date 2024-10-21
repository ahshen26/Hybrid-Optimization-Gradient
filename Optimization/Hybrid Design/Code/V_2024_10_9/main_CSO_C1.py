import pandas as pd
import Calibration1 as C1
import Calibration2 as C2
import Calibration3 as C3
import Calibration4 as C4
from mealpy.swarm_based.CSO import OriginalCSO
from mealpy import FloatVar
import numpy as np
import pickle

# Set-up Bounds

# Randomization ratio;
# Equivalence margin;
min_bound = np.array([0.1, 0])
max_bound = np.array([0.9, 1])

#############################################################################################
######################################### CSO ###############################################
#############################################################################################

# Calibration 1
def extract_cost1(input):
    # Call the original objective function
    _, _, cost = C1.fun_Power1(input)
    return cost

problem1 = {
    "obj_func": extract_cost1,    # Objective function
    "bounds": FloatVar(lb=min_bound, ub=max_bound),
    "minmax": "min",                   # Minimize the objective function
    "verbose": True                    # Print the progress
}

model1 = OriginalCSO(
    epoch=300,        # Number of iterations
    pop_size=100      # Population size
)

Results1 = model1.solve(problem=problem1)

with open('log/history_CSO_C1.pkl', 'wb') as f:
    pickle.dump(model1.history, f)

result1_CSO = C1.fun_Power1(Results1.solution)

combined_data = [int(200*Results1.solution[0])/200, Results1.solution[1]] + result1_CSO[1] + result1_CSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result1_CSO = pd.DataFrame([combined_data])
DF_result1_CSO.columns = ['Randomization Ratio', 'Equivalence Margin'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result1_CSO.to_csv('Results/DF_result1_CSO.csv', index=False)