import pandas as pd
import Sequential_C1 as C1
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
optimizer1_PSO = ps.single.GlobalBestPSO(n_particles=100, dimensions=3, options=options_PSO, bounds=(min_bound, max_bound))
# Perform optimization
cost1_PSO, pos1_PSO = optimizer1_PSO.optimize(C1.fun_Cost1, iters=200)

data_to_store = {
    'cost_history': optimizer1_PSO.cost_history,
    'pos_history': optimizer1_PSO.pos_history,
    'swarm': optimizer1_PSO.swarm
}
with open('log/sequential_history_PSO_C1.pkl', 'wb') as f:
    pickle.dump(data_to_store, f)

result1_PSO = C1.fun_Power1(pos1_PSO)

combined_data = [int(200*pos1_PSO[0])/200, int(200*pos1_PSO[0]*pos1_PSO[1])/int(200*pos1_PSO[0]), pos1_PSO[2]] + result1_PSO[1] + result1_PSO[0]
combined_data = [round(num, 4) for num in combined_data]
DF_result1_PSO = pd.DataFrame([combined_data])
DF_result1_PSO.columns = ['Split ratio between stage 1 and stage 2', 'Randomization Ratio', 'Equivalence Margin'] + [f'Power_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)] + [f'TypeIError_{round(i, 4)}' for i in np.arange(-0.6, 0.61, 0.05)]

DF_result1_PSO.to_csv('Results/Sequential_result1_PSO.csv', index=False)