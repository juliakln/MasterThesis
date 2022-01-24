import os
import numpy as np


def read_hist_exp(file):
    """ Read txt files containing histograms of stinging bees after the experiments
    Args:
        file: name of txt file that contains data
        First line: colony sizes, f.ex. [1 5 8 10]
        Then 1 line for each colony size with frequencies for each outcome, f.ex. [0.13 0.87] /n [0.2 0.1 0.3 0.2 0.1 0.1] ...
    Returns:
        colony_sizes: list of colony sizes 
        outputs: nested list with frequencies for each size
    """

    outputs = []
    
    with open((f"../data/{file}"), 'r') as f:
        data = f.read()

        # sizes in first line of txt file
        colony_sizes = np.array([int(c) for c in (data.split("\n")[0]).split(",")])

        y_frequencies = (data.split("\n")[1:])

        for f in y_frequencies:
            freq = np.array([float(i) for i in f.split(",")])
            outputs.append(list(freq))

    return colony_sizes, outputs


def read_stochnet(thresh, scale):
    """ Read txt files containing number of stinging bees after simulating CRN using stochnet
    and collect how often property is satisfied for trajectory
    """
    collect_data = {}

    for dirpath, dirs, files in os.walk("../data/stochnet"):
        for file in files:
            nbees = int((file.split("_")[1]).split(".")[0])
            with open(os.path.join(dirpath, file), 'r') as f:
                data = f.read()
                x_deadbees = (data.split("]")[0])[2:].replace(".","")
                y_frequencies = (data.split("]")[1])[2:]
                # compute number of living bees 
                bees = np.array([nbees-int(i) for i in x_deadbees.split()])
                freq = np.array([int(i) for i in y_frequencies.split()])
                
                threshold = np.ceil(thresh * nbees)            
                satisfactions = np.sum(freq[bees >= threshold])
                
                collect_data[nbees] = satisfactions
                
    # population size n together with number of trajectories satisfying property, sort by n
    paramValueSet = []
    paramValueOutputs = []
    for key in sorted(collect_data):
        paramValueSet.append(key)
        paramValueOutputs.append(collect_data[key])

    return np.array(paramValueSet).reshape(-1,1), (np.array(paramValueOutputs).reshape(-1,1))/scale
