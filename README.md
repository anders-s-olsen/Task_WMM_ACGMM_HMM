## ICASSP2023 submission on Watson and Angular central Gaussian mixture models and Hidden Markov models for modeling task fMRI data

This repository contains a set of Python and Matlab functions to generate the results presented in the conference paper:

>[Angular Central gaussian and Watson Mixture Models for Assessing Dynamic Functional Brain Connectivity During a Motor Task ](https://ieeexplore.ieee.org/document/10193021) 
Anders S. Olsen, Emil Ortvald, Kristoffer H. Madsen, Mikkel N. Schmidt, Morten Mørup.

The code is structured as follows: All source code including modules for Watson, ACG, mixture models, and HMM is located in src/models/. Preprocessing scripts is located in src/preprocessing/. Code for visualization is located in src/visualization/. All Python scripts used for executing the optimization of models is located in the experiments folder. The submitfiles/ folder exclusively contains shell scripts for submitting training jobs to DTU's compute cluster. 
