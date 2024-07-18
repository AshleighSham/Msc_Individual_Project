import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy

read_dictionary = np.load('test_data.npy',allow_pickle='TRUE').item()

print(read_dictionary['thetaj'])