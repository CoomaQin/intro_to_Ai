import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# a = np.zeros(16 * 16, dtype=int)
# print(a.reshape([16, 16]))

uniform_data = np.random.rand(16, 1)
print(uniform_data)
ax = sns.heatmap(uniform_data.reshape([4, 4]), linewidth=0.5)
plt.show()
