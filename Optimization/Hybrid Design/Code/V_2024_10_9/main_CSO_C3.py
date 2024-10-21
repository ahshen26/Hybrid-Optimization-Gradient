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
# Split Ratio
min_bound = np.array([0.1, 0, 0.01])
max_bound = np.array([0.9, 1, 0.99])

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
    epoch=300,        # Number of iterations
    pop_size=100,      # Population size
)

Results3 = model3.solve(problem=problem3)

with open('log/history_CSO_C3.pkl', 'wb') as f:
    pickle.dump(model3.history, f)

result3_CSO = C3.fun_Power3(Results3.solution)

combined_data = [int(200*Results3.solution[0])/200, Results3.solution[1], Results3.solution[2]] + result3_CSO[1] + result3_CSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result3_CSO = pd.DataFrame([combined_data])
DF_result3_CSO.columns = ['Randomization Ratio', 'Equivalence Margin', 'Split Proportion'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result3_CSO.to_csv('Results/DF_result3_CSO.csv', index=False)