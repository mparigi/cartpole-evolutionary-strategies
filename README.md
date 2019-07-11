# Evolutionary Strategies for Reinforcement Learning: CartPole


## Running

```
python3 es-cartpole.py
```
The above command will run the program.

## Algorithm Explanation

This program uses Lambda-Mu Evolutionary Strategies in combination with a Neural Network with no hidden layers
(essentially a linear combination of the inputs) to make an action decision at each time step.

The evolution process is as follows:

```
Generate population of lambda random agents
Evaluate each agent by running it
Get the mu best scoring agents (mu must divide lambda)
These mu agents ("parents") each produce lambda / mu "children"
  To produce a child, the "parent" is blurred with random noise
The children become the new population
```
Note that this is not Lambda+Mu ES, which is elitist.

An agent makes decisions using the following policy:

```
We suppose the observation has N real values ([o_1, ... , o_N]).
The agent has N real valued weights ([w_1, ... , w_N]).
The agent also has a real valued bias (b).
An action decision is made by linearly combining the weights/bias and the observation,
  then squishing via sigmoid to (0,1), then stretching this range out to the number
  of possible actions (A), then mapping this continuous space to the discrete space of actions
  via flooring.

Stated mathematically,

action = floor(A * σ(w_1*o_1 + w_2*o_2 + ... + w_N*o_N + b))

This weight/bias to output scenario can be viewed as a neural network from the inputs directly to a single output
(which is then mapped to the discrete action space) with no hidden layers.

```

## Output Explanation

Every generation, the scores of each individual of the population is printed,
  along with the score of the best performing individual, and the average score
  across the population.
  
At the end of the program, a graph is displayed of the *best* score of each population over the generations.
