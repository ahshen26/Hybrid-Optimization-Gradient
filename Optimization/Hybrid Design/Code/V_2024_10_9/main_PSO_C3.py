import pandas as pd
import Calibration1 as C1
import Calibration2 as C2
import Calibration3 as C3
import Calibration4 as C4
import pyswarms as ps
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
######################################### PSO ###############################################
#############################################################################################
# Set-up Hyperparameters
options_PSO = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Call instance of PSO
optimizer3_PSO = ps.single.GlobalBestPSO(n_particles=100, dimensions=3, options=options_PSO, bounds=(min_bound, max_bound))
# Perform optimization
cost3_PSO, pos3_PSO = optimizer3_PSO.optimize(C3.fun_Cost3, iters=300)

data_to_store = {
    'cost_history': optimizer3_PSO.cost_history,
    'pos_history': optimizer3_PSO.pos_history,
    'swarm': optimizer3_PSO.swarm
}
with open('log/history_PSO_C3.pkl', 'wb') as f:
    pickle.dump(data_to_store, f)

result3_PSO = C3.fun_Power3(pos3_PSO)

combined_data = [int(200*pos3_PSO[0])/200, pos3_PSO[1], pos3_PSO[2]] + result3_PSO[1] + result3_PSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result3_PSO = pd.DataFrame([combined_data])
DF_result3_PSO.columns = ['Randomization Ratio', 'Equivalence Margin', 'Split Proportion'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result3_PSO.to_csv('Results/DF_result3_PSO.csv', index=False)