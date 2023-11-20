import numpy as np
import matplotlib.pyplot as plt
vals = np.linspace(0,1,256)
np.random.shuffle(vals)
cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))

print(cmap)