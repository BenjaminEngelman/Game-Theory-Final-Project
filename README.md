# Game-Theory-Final-Project

## Outlook

### algos.py

Contains the implementation of all the RL algorithm. Each algorithm has two version :
  - One version using Tables for the value-functions
  - One version using neural networks to approximate the values of the value-functions
  
### agent.py

Contains the implementation of the agent used in the 5 experiments

### experimentConf.py

Contains all the parameters that we used for the algorithms/ensemble methods for each experiment

### mazes.py

Contains the implementations off all the environments (mazes) needed for the 5 experiments as well as functions to generate such environments.

### clusterRunner.py

Script used to run the experiments on a cluster and distribute the trials across multiple cores. 


## TODO

### Neural nets

  - [x] Q-learning
  - [x] SARSA
  - [x] Actor-Critic
  - [x] QV-learning
  - [x] ACLA

### Belief State

  - [x] Belief State
  - [x] Maze observations


### Experiments

  - [ ] Exp 1 (Simple maze + base algo)
  - [ ] Exp 2 (Partially obsebable maze + neural net)
  - [ ] Exp 3 (Dynamic obstacles maze + neural net)
  - [ ] Exp 4 (Dynamic Goal maze + neural net) 
  - [ ] Exp 5 (Generalized maze + neural net)

### Base Algorithms

  - [x] Q-learning
  - [x] SARSA
  - [x] Actor-Critic
  - [x] QV-learning
  - [x] ACLA
 
 ### Ensemble methods
 
  - [x] Majority voting
  - [x] Rank voting
  - [x] Boltzmann multiplication
  - [x] Boltzmann addition

### Environments
  
  - [x] Simple Dyna maze (9x6)
  - [x] Dyna maze with Dynamic Goal (9x6)
  - [x] Dyna maze with dynamic obstacles (9x6)
  - [x] Generalized maze (9x6)
