import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elasticity_2D\EnKF.csv')

print(data['E'])

fig, ax = plt.subplots()
ax.plot(range(len(data['E'])), data['E'])

plt.show()