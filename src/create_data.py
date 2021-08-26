"""
Create data to test methodology

1. Single value output for n = 2, 3, 4, 5, 7, 10 bees


2. Histogram output for n = 2, 3, 5, 10 bees
    Follows normal distribution
"""

import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats



# baseline for single value output: sinus function
def sinus_func(x):
    return np.sin(2*x) + (0.6*x)

def create_single_output():
    """ Create baseline function and take values for n=2,3,4,5,7,10 bees

    Returns:
    x_true : all training x values
    y_true : all training y values
    x_data : data x values (colony sizes)
    y_data : data y values  
    """
    x_true = np.linspace(0,10,50)
    x_data = np.array([2,3,4,5,7,10])

    y_true = sinus_func(x_true)
    y_data = sinus_func(x_data)

    return [x_true, y_true], [x_data, y_data]

def plot_single_output():
    [xt,yt], [xd,yd] = create_single_output()
    plt.figure(figsize=(12,6))
    plt.plot(xt, yt, label='True Function')
    plt.plot(xd, yd, 'r.', label='True Data Points')
    plt.xlabel('Colony size')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.savefig('figures/data/single_output.png')

def print_single_output():
    _, [xd,yd] = create_single_output()
    with open('data/single_output.txt', 'w') as f:
        for item in [xd,yd]:
            f.write("%s\n" % item)



#single_value_output = [0.5, 1.7, 7.4, 3.9]
#plt.plot(colony_sizes, single_value_output, 'b.')
#plt.show()

#with open('data/single_output.txt', 'w') as f:
#    for item in single_value_output:
#        f.write("%s\n" % item)


"""
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

"""


def main():
    plot_single_output()
    print_single_output()


if __name__ == "__main__":
    sys.exit(main())