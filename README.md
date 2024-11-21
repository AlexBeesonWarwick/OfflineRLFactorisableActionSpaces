# An investigation of offline reinforcement learning in factorisable action spaces

This repository contains the code and datasets used to produce the results in our [paper published in TMLR](https://openreview.net/pdf?id=STwxyUfpNV).

Installation instructions for the DeepMind Control suite can be found [here](https://github.com/google-deepmind/dm_control).

Installation instructions for the Maze environment can be found [here](https://github.com/yashchandak/lifelong_changing_actions).

Datasets can be downloaded from [here][https://warwickfiles.warwick.ac.uk/s/GrGH9RsyDRajASq?path=%2F].

## Instructions for running code
We provide individual examples of running each algorithm for one set of DMC and Maze datasets.  To train on a different dataset, simply update the associated parameters.

### DMC

For the DMC suite we have created a separate package for loading environments and datasets.  This can be installed by cloning this repository, navigating to the root and running 
```
pip install -r requirements.txt
```
Further instructions, including folder structures for the data, can be accessed [here](https://github.com/davidireland3/dmc_datasets).

### Maze

Be sure to update expert and random scores as well as the number of sub-action dimensions.

A full list of expert and random scores is available in the file "Expert_Random_Scores.csv"

## Feedback 
If you experience any problems or have any queries, please raise an issue or pull request.
