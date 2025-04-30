# ALPAI_proj

## Setup
- Clone the repo
```
git clone git@github.com:BabaYaga840/ALPAI_proj.git
```
- Use environment.yml to create the conda env
```
conda env create -f environment.yml
```
- Run A*
```
python astar.py
```

## scripts
Contains the path planning and environmnt files

### Environemnt scripts
#### env.py
Simple gridworld environment with uncertainity map (Do not use)
#### dynamic_env.py
Dynamic gridworld environment with uncertainity map 
- Dynamic changes can be toggled off or on
- Initial observation can be set to true map state or just local observation (allowing for blind exploration)

### Path planning scripts
#### astar.py
Implementation of simple A* for the Dynamic Environment
#### dqn.py
Implementation of Double DQN
#### bc_agent.py
Implementation of BC agent using A* to generate expert data
