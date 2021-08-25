"""
Create data to test methodology

1. Single value output for n = 2, 3, 5, 10 bees


2. Histogram output for n = 2, 3, 5, 10 bees
    Follows normal distribution
"""
import sys
import os
sys.path.insert(0, "C:/Users/klein/Documents/Uni/MasterThesis/")
os.chdir("C:/Users/klein/Documents/Uni/MasterThesis/")
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


colony_sizes = [2, 3, 5, 10]

# create random single value output and save in txt file
single_value_output = [0.5, 1.7, 7.4, 3.9]

plt.plot(colony_sizes, single_value_output, 'b.')
plt.show()

with open('data/single_output.txt', 'w') as f:
    for item in single_value_output:
        f.write("%s\n" % item)


# create truncated normal distribution for histogram output for all colony sizes
outputs = []
for size in colony_sizes:
    mu, sigma = size/2, size/4
    lower, upper = 0, size
    x = scipy.stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)
    plt.hist(x.rvs(10000), density=True, bins=np.linspace(-0.5,size+0.5,size+2))
    y, x = np.histogram(x.rvs(10000), bins=np.linspace(-0.5,size+0.5,size+2), density=True)
    print(f'Results for {size} bees:', y)
    outputs.append(y)
    plt.show()

with open('data/histogram_outputs.txt', 'w') as f:
    for item in outputs:
        f.write("%s\n" % item)