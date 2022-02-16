"""
Read data files from simulated or experimental data
"""

import os
import sys
import numpy as np
import subprocess
from collections import defaultdict


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

def read_exp_smmc(col, out, t):
    """ 
    Read satisfaction probability of experimental data from histograms

    Args:
        col: list of colony sizes
        out: nested list of frequency outputs after experiment
        t: threshold for "min. t bees are alive after experiment" (fraction)
    Returns:
        training input and output
    """
    paramValueSet = np.array(col).reshape(-1,1)
    satisfactions = []
    for i in np.arange(len(col)):
        # compute number of living bees
        bees = list(reversed(out[i]))
        # compute threshold = how many bees do at least have to be alive?
        threshold = int(np.ceil(t * col[i]))
        # sum up the frequencies for all outcomes of experiment that satisfy the property
        satisfactions.append(np.sum(bees[threshold:]))
    paramValueOutput = np.array(satisfactions).reshape(-1,1)

    return paramValueSet, paramValueOutput
    


def read_stochnet_hist(scale):
    """ Read txt files containing number of stinging bees after simulating CRN using stochnet
    to create histogram frequencies used for GPR
    """
    collect_data = {}
    colony_sizes = []

    for dirpath, dirs, files in os.walk("../data/stochnet"):
        for file in files:
            nbees = int((file.split("_")[1]).split(".")[0])
            #count_helper = {}
            with open(os.path.join(dirpath, file), 'r') as f:
                data = f.read()
                x_deadbees = (data.split("]")[0])[2:].replace(".","")
                y_frequencies = (data.split("]")[1])[2:]

                deadbees = [int(i) for i in x_deadbees.split()]
                freq = [int(i)/scale for i in y_frequencies.split()]

                count_helper = dict.fromkeys(np.arange(nbees+1), 0)
                for d, f in zip(deadbees, freq):
                    count_helper[d] = f

                outputs_helper = []
                for key in sorted(count_helper):
                    outputs_helper.append(count_helper[key])
                
                collect_data[nbees] = outputs_helper
                
    # population size n together with number of trajectories satisfying property, sort by n
    paramValueSet = []
    outputs = []
    for key in sorted(collect_data):
        colony_sizes.append(key)
        outputs.append(collect_data[key])

    return np.array(colony_sizes), outputs



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



def read_stochnet2(thresh, scale):
    """ Data for 2 dimensions -> vary N and one of the rates k
    Read txt files containing number of stinging bees after simulating CRN using stochnet
    and collect how often property is satisfied for trajectory
    """
    collect_data = {}

    for dirpath, dirs, files in os.walk("../data/stochnet2"):
        for file in files:
            if file.startswith("bees"):
                nbees = int((file.split("_")[1]).split(".")[0])
                k = round(float((file.split("_")[2]).rsplit(".", 1)[0]), 4)
                with open(os.path.join(dirpath, file), 'r') as f:
                    data = f.read()
                    x_deadbees = (data.split("]")[0])[1:].replace(".","")
                    y_frequencies = (data.split("]")[1])[2:]
                    # compute number of living bees 
                    bees = np.array([nbees-int(i) for i in x_deadbees.split()])
                    freq = np.array([int(i) for i in y_frequencies.split()])
                    
                    threshold = np.ceil(thresh * nbees)            
                    satisfactions = np.sum(freq[bees >= threshold])
                    
                    collect_data[(nbees,k)] = satisfactions
                    
    # population size n together with number of trajectories satisfying property, sort by n
    paramValueSet = np.zeros((len(collect_data), 2))
    paramValueOutputs = []
    i = 0
    for key in sorted(collect_data):
        paramValueSet[i,] = key
        paramValueOutputs.append(collect_data[key])
        i += 1

    return paramValueSet, (np.array(paramValueOutputs).reshape(-1,1))/scale


def simulate_bee_prism(paths):
    # PRISM - ONLY RUN TO SIMULATE NEW DATA
    # uncertain parameter values for training data
    p1 = np.linspace(0, 1, 15) 

    # simulate chain with Prism and save paths
    for p in p1:
        result = "/Users/juliaklein/Documents/uni/MasterThesis/data/dtmc_1/case_" + str(p) + ".txt"
        resultfile = open(result ,"w")
        resultfile.close()
        resultfile = open(result, "r+")
        prismcommand = "/Applications/prism-4.7-src/prism/bin/prism /Users/juliaklein/Documents/uni/MasterThesis/models/bee_3.pm /Users/juliaklein/Documents/uni/MasterThesis/models/bee_3_p.pctl -const p=" + str(p) + " -sim -simsamples " + str(paths) + " -exportresults " + result
        prismprocess = subprocess.check_call(prismcommand, stdin=None, stdout=None , stderr=None, shell=True)
        resultfile.close()

def simulate_bee_prism_pmc():
    p1 = np.linspace(0, 1, 100) 
    # simulate chain with Prism and save paths
    for p in p1:
        result = "/Users/juliaklein/Documents/uni/MasterThesis/data/dtmc_1_pmc/case_" + str(p) + ".txt"
        resultfile = open(result ,"w")
        resultfile.close()
        resultfile = open(result, "r+")
        prismcommand = "/Applications/prism-4.7-src/prism/bin/prism /Users/juliaklein/Documents/uni/MasterThesis/models/bee_3.pm /Users/juliaklein/Documents/uni/MasterThesis/models/bee_3_p.pctl -const p=" + str(p) + " -exportresults " + result
        prismprocess = subprocess.check_call(prismcommand, stdin=None, stdout=None , stderr=None, shell=True)
        resultfile.close()

def simulate_bee_prism2(paths):
    # PRISM - ONLY RUN TO SIMULATE NEW DATA
    # uncertain parameter values for training data
    X = np.zeros((100,2))
    p1 = np.linspace(0.1, 1, 10) 
    p2 = np.linspace(0.1, 1, 10)
    X[:,0] = np.repeat(p1, 10)
    X[:,1] = np.tile(p2, 10)
    # simulate chain with Prism and save paths
    for x in X:
        p1 = x[0]
        p2 = x[1]
        result = "/Users/juliaklein/Documents/uni/MasterThesis/data/dtmc_2/case_" + str(round(p1,1)) + "_" + str(round(p2,2)) + ".txt"
        resultfile = open(result ,"w")
        resultfile.close()
        resultfile = open(result, "r+")
        prismcommand = "/Applications/prism-4.7-src/prism/bin/prism /Users/juliaklein/Documents/uni/MasterThesis/models/bee_3.pm /Users/juliaklein/Documents/uni/MasterThesis/models/bee_3_p.pctl -const p=" + str(p1) + ",q1=" + str(p2) + " -sim -simsamples " + str(paths) + " -exportresults " + result
        prismprocess = subprocess.check_call(prismcommand, stdin=None, stdout=None , stderr=None, shell=True)
        resultfile.close()


def read_bee_prism():
    # save number of satisfactions for each value of p and experiment
    satisfactions = defaultdict(list)
    # read outcomes for all parameter values and compute number of satisfactions
    for dirpath, dirs, files in os.walk("/Users/juliaklein/Documents/uni/MasterThesis/data/dtmc_1"):
        for file in files:
            if file.startswith("case"):
                p = round(float((file.split("_")[1]).rsplit(".", 1)[0]), 4)
                with open(os.path.join(dirpath, file), 'r') as f:
                    data = f.readline()
                    for last_line in f:
                        pass
                    S = float(last_line)
                    satisfactions[p].append(S)

    # Training data
    paramValueSet = []
    paramValueOutputs = []
    for key in sorted(satisfactions):
        paramValueSet.append(key)
        paramValueOutputs.append(satisfactions[key])

    return np.array(paramValueSet).reshape(-1,1), (np.array(paramValueOutputs).reshape(-1,1))

def read_bee_prism_pmc():
    # save probability of satisfactions for each population size and experiment
    pmc_satisfactions = defaultdict(list)
    # read outcomes for all population sizes and compute number of satisfactions
    for dirpath, dirs, files in os.walk("/Users/juliaklein/Documents/uni/MasterThesis/data/dtmc_1_pmc"):
        for file in files:
            if file.startswith("case"):
                p = round(float((file.split("_")[1]).rsplit(".", 1)[0]), 4)
                with open(os.path.join(dirpath, file), 'r') as f:
                    data = f.readline()
                    for last_line in f:
                        pass
                    S = float(last_line)
                    pmc_satisfactions[p].append(S)
    # PMC data
    pmc_X = []
    pmc_f = []
    for key in sorted(pmc_satisfactions):
        pmc_X.append(key)
        pmc_f.append(pmc_satisfactions[key])
        
    return np.array(pmc_X).reshape(-1,1), np.array(pmc_f).reshape(-1,1)


def read_bee_prism2():
    """ Data for 2 dimensions -> vary N and one of the rates k
    Read txt files containing number of stinging bees after simulating CRN using stochnet
    and collect how often property is satisfied for trajectory
    """
    satisfactions = defaultdict(list)

    for dirpath, dirs, files in os.walk("../data/dtmc_2"):
        for file in files:
            if file.startswith("case"):
                p1 = float((file.split("_")[1]))
                p2 = float((file.split("_")[2]).rsplit(".", 1)[0])
                with open(os.path.join(dirpath, file), 'r') as f:
                    data = f.readline()
                    for last_line in f:
                        pass
                    S = float(last_line)
                    satisfactions[(p1, p2)] = S
                    
    # population size n together with number of trajectories satisfying property, sort by n
    paramValueSet = np.zeros((len(satisfactions), 2))
    paramValueOutputs = []
    i = 0
    for key in sorted(satisfactions):
        paramValueSet[i,] = key
        paramValueOutputs.append(satisfactions[key])
        i += 1

    return paramValueSet, (np.array(paramValueOutputs).reshape(-1,1))




def main():
    #simulate_bee_prism(50)
    #simulate_bee_prism2(50)
    simulate_bee_prism_pmc()

if __name__ == "__main__":
    sys.exit(main())