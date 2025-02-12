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
# Split Ratio
min_bound = np.array([0.1, 0, 0.01])
max_bound = np.array([0.9, 1, 0.09])

#############################################################################################
######################################## CSOMA ##############################################
#############################################################################################
# Set-up Hyperparameters
options_CSOMA = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'phi': 0.2}

# Call instance of PSO
optimizer3_CSOMA = CSOMA_Python.single.CSOMA(n_particles=100, dimensions=3, options=options_CSOMA, bounds=(min_bound, max_bound))
# Perform optimization
cost3_CSOMA, pos3_CSOMA = optimizer3_CSOMA.optimize(C3.fun_Cost3, iters=300)

data_to_store = {
    'cost_history': optimizer3_CSOMA.cost_history,
    'pos_history': optimizer3_CSOMA.pos_history,
    'swarm': optimizer3_CSOMA.swarm
}
with open('log/history_CSOMA_C3.pkl', 'wb') as f:
    pickle.dump(data_to_store, f)

result3_CSOMA = C3.fun_Power3(pos3_CSOMA)

combined_data = [int(200*pos3_CSOMA[0])/200, pos3_CSOMA[1], pos3_CSOMA[2]] + result3_CSOMA[1] + result3_CSOMA[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result3_CSOMA = pd.DataFrame([combined_data])
DF_result3_CSOMA.columns = ['Randomization Ratio', 'Equivalence Margin', 'Split Proportion'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result3_CSOMA.to_csv('Results/DF_result3_CSOMA.csv', index=False)