import Cost_Hybrid
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

x = Cost_Hybrid.fun_Power(np.array([0.5,0.2908]), 194, 0, 1,0.3962035, 100, 1, 0.05, 0.2, 4)

r = 0.5
n=185
alpha = 0.05
x1_var = 1 ** 2 / (n * r * (1 - r))
norm.cdf(-norm.ppf(1 - alpha / 2) * np.sqrt(x1_var), 0.3962035, np.sqrt(x1_var)) + (1 - norm.cdf(norm.ppf(1 - alpha / 2) * np.sqrt(x1_var), 0.3962035, np.sqrt(x1_var)))

df = pd.DataFrame({
    'Power': x[1],
    'type I error': x[0]
})
df = df.T
df.to_csv('optimalDesign_1_1_194.csv', index=False)
x[len(x) // 2]

x = np.arange(-1, 1.01, 0.01)  # 100 points between 0 and 1
power = results[1]
typeIError = results[0]
# Create the plot
plt.plot(x, power)
plt.xlabel('Difference in mean between RWD and RCT control arm')
plt.ylabel('Power')
plt.title('Impact of Bias on Power; (r = 1:1, m = 0.2879, N = 193)')
plt.legend()
plt.grid(True)
plt.show()
# Create the plot
plt.plot(x, typeIError)
plt.xlabel('Difference in mean between RWD and RCT control arm')
plt.ylabel('Type I Error')
plt.title('Impact of Bias on Type I Error; (r = 1:1, m = 0.2879, N = 193)')
plt.legend()
plt.grid(True)
plt.show()