# POMDP Decision Theory Application
This application was made to help create 

## Set Up

1. Clone the repository via git
2. ensure the requirements.txt is downloaded
3. run the bash files or the single config yaml files in the 

## Application

This applicaiton is intended to do 2 things

1. Given a support 
2. Experiment, analyze and compare between different decision theory problems
2. Create stochastic systems/markov systems and formally verify properties from specified decision theory paradigms/ 
3. be able to construct DTMC/CTMC from the given agent trajectoreis to then input into formal verificaiton algorithms

## Goals

- **Verify Active Inference Properties:**
  - Active Inference has the potential to model the generative process for how intelligent biological organism interact with their external environments
  - A key goal in this project is to specify and then verify properties within Active Inference systems
- **Comparison with Reinforcement Learning and PRISM**

## Supported Experiments

### 2D Gridworld

This enviornment repsentes a 2D Gridworld. Th following set up occurs

- an agent is spawned in a rndom location
- the agent is goal to is to reach the given goal
- the agent is preprogrammed to observe the state its in, a possible boudnary and reward observation
- the agent spawns in a random loaction in the grid

#### Extensions

##### Environment Complexity

**Environment Complexity**

There are several possible ways to make the environment more complicated

- **Blocks**
  - Random blocks are added across the gird
  - The agent can observe them, but not move within these given blocks
- **Cues**
  - Cues provide the agent with the ability to do temporal planning, where it makes multiple mini strategies towards its goal
- **Negative Rewards**
  - There are traps that the agent learns to avoid, that if activated, will reset things

**Stochastic Environment Properties**

There are possible ways to make the environment more complicated via stochastic processes
- the agents transition and observation model can change by a significant amount in random aspects
- used to experiment the agents ability to respond to robust environment changes
- one can specify the degree of offset of the observation and transition model

##### Agent Complexity

**Set Partial Observability/Actions**

- The agents staring partial observability can be varied
- This can include full observability
- An offset of partiality which varies from 0 to 100%
- Randomly distributed generated distributions



### Supported Environment

####  nxn grid, where agent need to reach a goal

**Base Experiemnt**

##### nxn grid with maze, where agent needs to reach goal

### Supported Agents

- Active Inference 
  - PYMDP
- Reinforcement Learning 
  - RNN-PPO
  - Q-Learning Based 