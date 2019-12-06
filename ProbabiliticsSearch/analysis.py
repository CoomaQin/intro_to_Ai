import landscape as map
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

h = map.play_rule1(30)
# h = map.play_rule2(30)
# h = map.play_moving(30)
ax = sns.heatmap(h.reshape([30, 30]), linewidth=0.5)
plt.show()

