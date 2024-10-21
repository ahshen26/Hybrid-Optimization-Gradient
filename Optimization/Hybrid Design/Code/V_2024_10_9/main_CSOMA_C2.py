import pandas as pd
import Calibration1 as C1
import Calibration2 as C2
import Calibration3 as C3
import Calibration4 as C4
import CSOMA_Python
import numpy as np
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history
import pickle


# Set-up Bounds

# Randomization ratio;
# Equivalence margin;
min_bound = np.array([0.1, 0])
max_bound = np.array([0.9, 1])

#############################################################################################
######################################## CSOMA ##############################################
#############################################################################################
# Set-up Hyperparameters
options_CSOMA = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'phi': 0.2}

# Call instance of PSO
optimizer2_CSOMA = CSOMA_Python.single.CSOMA(n_particles=100, dimensions=2, options=options_CSOMA, bounds=(min_bound, max_bound))
# Perform optimization
cost2_CSOMA, pos2_CSOMA = optimizer2_CSOMA.optimize(C2.fun_Cost2, iters=300)

data_to_store = {
    'cost_history': optimizer2_CSOMA.cost_history,
    'pos_history': optimizer2_CSOMA.pos_history,
    'swarm': optimizer2_CSOMA.swarm
}
with open('log/history_CSOMA_C2.pkl', 'wb') as f:
    pickle.dump(data_to_store, f)

result2_CSOMA = C2.fun_Power2(pos2_CSOMA)

combined_data = [int(200*pos2_CSOMA[0])/200, pos2_CSOMA[1]] + result2_CSOMA[1] + result2_CSOMA[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result2_CSOMA = pd.DataFrame([combined_data])
DF_result2_CSOMA.columns = ['Randomization Ratio', 'Equivalence Margin'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result2_CSOMA.to_csv('Results/DF_result2_CSOMA.csv', index=False)