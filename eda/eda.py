import numpy as np
from eda.graphs import bar_chart_ylabel, show_images_around_index
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data1 = np.load(BASE_DIR / "data/X_train.npy")
data2 = np.load(BASE_DIR / "data/Y_train.npy")

print("shape of data X_train:")
print(data1.shape)
print("shape of data Y_train:")
print(data2.shape)

bar_chart_ylabel(data2).show()

show_images_around_index(data1, data2, 4000).show()
