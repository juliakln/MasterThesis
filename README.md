# MasterThesis: Applications of Gaussian Processes in Modelling Biological Collectives

Combine formal verification with Gaussian Processes, a powerful Machine Learning approach, to develop a scalable, flexible, and data-efficient method in the analysis of biological collectives. We address the following research questions:

- predict the collective response of animal populations of different sizes
- derive the fitness function a collective tries to optimise, and
- infer parameters of an uncertain stochastic model to model a desired behaviour.

The application of the developed frameworks is shown on a case study analysing a social feedback mechanism in honeybees [1] within three experimental contexts.
This repository is structured as follows:


## notebooks

Three Jupyter Notebooks implement methods used throughout the thesis: Gaussian Process Regression [2], Smoothed Model Checking [3], and two-dimensional Smoothed Model Checking. The implementations are supported by detailed theoretical backgrounds, and demonstrated on a running example, the well-known SIR model [4].  
Furthermore, one Notebook illustrates the relationship between stochastic models and determinstic models, as a short general remark.


## models

For the case study, we analyse different experimental contexts. Besides experimental data, we simulate a Chemical Reaction Network (CRN) using the tool StochNetV2 [5], and analyse a Discrete-Time Markov Chain using PRISM [6]. The corresponding models, properties and scripts to run the tools can be found in this folder.


## data

This folder contains txt files of experimental data, simulated data from a CRN, and simulated data from Markov Chains. These files are used as input to the developed frameworks. 


## figures

All results are visualised and saved. 


## src

This folder contains the main implementation in Python 3.9.7. One script runs analyses based on Gaussian Process Regression, and a different one based on one- or two-dimensional Smoothed Model Checking. Furthermore, we implemented different kernel functions, and a helper script to deal with the different input types of training data.


Other files are used only for the purpose of the Master Thesis. 



------------------------------------------------------------------------------


## References

[1] Matej Hajnal, Morgane Nouvian, David Šafranek, and Tatjana Petrov. Data-
Informed Parameter Synthesis for Population Markov Chains. In International
Workshop on Hybrid Systems Biology, pages 147–164, Cham, 2019. Springer.
doi: 10.1007/978-3-030-28042-0_10.  
[2] Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian processes
for machine learning, volume 2 of 3. MIT Press, Cambridge, MA, 2006.  
[3] Luca Bortolussi, Dimitrios Milios, and Guido Sanguinetti. Smoothed model
checking for uncertain continuous-time Markov chains. Information and Computation,
247:235–253, 2016.  
[4] William Ogilvy Kermack and Anderson G. McKendrick. A contribution to the
mathematical theory of epidemics. Proceedings of the Royal Society of London.
Series A, Containing Papers of a Mathematical and Physical Character, 115
(772):700–721, 1927. doi: 10.1098/rspa.1927.0118.  
[5] Denis Repin, Nhat-Huy Phung, and Tatjana Petrov. StochNetV2: A Tool
for Automated Deep Abstractions for Stochastic Reaction Networks. In International
Conference on Quantitative Evaluation of Systems, pages 27–32,
Cham, 2020. Springer.  
[6] Marta Kwiatkowska, Gethin Norman, and David Parker. PRISM 4.0: Verification
of probabilistic real-time systems. In International conference on computer
aided verification, pages 585–591, Berlin, Heidelberg, 2011. Springer.
