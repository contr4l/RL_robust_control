import cubic_spline_planner
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import argparse
import numpy as np
import re
from PIL import Image


# parameters
dt = 0.1  # time tick[s]
R = 50.  # visual trace circle R
v_head = 20/3.6
X_pos = np.linspace(-R, R, 200)
Y_pos = np.sqrt(R**2 - X_pos**2)
X_neg = np.linspace(R, -R, 200)
Y_neg = -np.sqrt(R**2 - X_neg**2)
X = np.hstack([X_pos, X_neg])
Y = np.hstack([Y_pos, Y_neg])

plt.figure(figsize=(5,5))
plt.plot(X, Y, "-g", linewidth=15)
plt.plot(X, Y, "-k", linewidth=1)
# plt.plot(1.02*X, 1.02*Y, "--k", linewidth=1)
# plt.plot(0.98*X, 0.98*Y, "--k", linewidth=1)
plt.show()
