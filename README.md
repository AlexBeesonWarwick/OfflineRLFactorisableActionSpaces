# An investigation of offline reinforcement learning in factorisable action spaces

This repository contains the code and datasets used to produce the results in our paper published in TMLR - https://openreview.net/pdf?id=STwxyUfpNV.

The benchmark datasets can be accessed here - https://warwickfiles.warwick.ac.uk/s/GrGH9RsyDRajASq

The algorithms used in the work can be found in the folder "Algorithms".

Our work makes use of a Maze environment and the DeepMind Control (DMC) suite.  Installation instructions for the Maze environment can be found here - https://github.com/yashchandak/lifelong_changing_actions.  Installation instructions for the DMC can be found here - https://github.com/google-deepmind/dm_control

## Instructions for running code
We provide individual examples of running each algorithm for one set of Maze and DMC datasets.  To train on a different dataset, simply update the environment parameters and file location.  Be sure to update expert and random scores as well as the number of sub-action dimensions.

A full list of expert and random scores is available in the file "Expert_Random_Scores.csv"

## Feedback 
If you experience any problems or have any queries, please raise an issue or pull request.
