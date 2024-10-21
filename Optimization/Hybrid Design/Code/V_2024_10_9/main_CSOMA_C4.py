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
optimizer4_CSOMA = CSOMA_Python.single.CSOMA(n_particles=100, dimensions=2, options=options_CSOMA, bounds=(min_bound, max_bound))
# Perform optimization
cost4_CSOMA, pos4_CSOMA = optimizer4_CSOMA.optimize(C4.fun_Cost4, iters=300)

data_to_store = {
    'cost_history': optimizer4_CSOMA.cost_history,
    'pos_history': optimizer4_CSOMA.pos_history,
    'swarm': optimizer4_CSOMA.swarm
}
with open('log/history_CSOMA_C4.pkl', 'wb') as f:
    pickle.dump(data_to_store, f)

result4_CSOMA = C4.fun_Power4(pos4_CSOMA)

combined_data = [int(200*pos4_CSOMA[0])/200, pos4_CSOMA[1]] + result4_CSOMA[1] + result4_CSOMA[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result4_CSOMA = pd.DataFrame([combined_data])
DF_result4_CSOMA.columns = ['Randomization Ratio', 'Equivalence Margin'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result4_CSOMA.to_csv('Results/DF_result4_CSOMA.csv', index=False)