# Imports
import numpy as np
from eda.graphs import *

# Data loading
data1 = np.load("dc1/data/X_train.npy")
data2 = np.load("dc1/data/Y_train.npy")

# Data shape visualization.
print("shape of data X_train:")
print(data1.shape)
print("shape of data Y_train:")
print(data2.shape)

# Y-label data count visualization.
bar_chart_ylabel(data2).show()

# Coustome index image viewing with label
show_images_around_index(data1, data2, 4000).show()
