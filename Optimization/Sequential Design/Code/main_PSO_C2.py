import pandas as pd
import Sequential_C2 as C2
import pyswarms as ps
import numpy as np
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history
import pickle

# Set-up Bounds

# Split ratio between stage 1 and stage 2
# Randomization ratio;
# Equivalence margin;
# Split proportion of type I error rate
min_bound = np.array([0.1, 0.1, 0])
max_bound = np.array([0.9, 0.9, 1])

#############################################################################################
######################################### PSO ###############################################
#############################################################################################
# Set-up Hyperparameters
options_PSO = {'c1': 0.5, 'c2': 0.3, 'w': 0.7}

# Call instance of PSO
optimizer2_PSO = ps.single.GlobalBestPSO(n_particles=100, dimensions=3, options=options_PSO, bounds=(min_bound, max_bound))
# Perform optimization
cost2_PSO, pos2_PSO = optimizer2_PSO.optimize(C2.fun_Cost2, iters=200)

data_to_store = {
    'cost_history': optimizer2_PSO.cost_history,
    'pos_history': optimizer2_PSO.pos_history,
    'swarm': optimizer2_PSO.swarm
}
with open('log/sequential_history_PSO_C2.pkl', 'wb') as f:
    pickle.dump(data_to_store, f)

result2_PSO = C2.fun_Power2(pos2_PSO)

combined_data = [int(200*pos2_PSO[0])/200, int(200*pos2_PSO[0]*pos2_PSO[1])/int(200*pos2_PSO[0]), pos2_PSO[2]] + result2_PSO[1] + result2_PSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result2_PSO = pd.DataFrame([combined_data])
DF_result2_PSO.columns = ['Split ratio between stage 1 and stage 2', 'Randomization Ratio', 'Equivalence Margin'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result2_PSO.to_csv('Results/Sequential_result2_PSO.csv', index=False)