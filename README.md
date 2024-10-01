# ReInforcementLearningSamp
Reinforcement Learning Samples
Reinforcement learning problems are sequential decision-making problems.
Ex., Autonomous vehicles, such as self-driving cars and aerial drones. Robot-arm manipulation tasks, such as removing a nail with a hammer
S is the set of all possible states.
A is the set of all possible actions.
R is the distribution of reward given a state-action pair—some particular state paired with some particular action—denoted as (s, a). It’s a distribution in the sense of being a probability distribution: The exact same state-action pair (s, a) might randomly result in different amounts of reward r on different occasions.
P, like R, is also a probability distribution. In this case, it represents the probability of the next state (i.e., st+1) given a particular state-action pair (s, a) in the current timestep t. Like R, the P distribution is hidden from the agent, but again aspects of it can be inferred by taking actions within the environment. 
γ (gamma) is a hyperparameter called the discount factor (also known as decay). To explain its significance, when the agent considers the value of a prospective reward, it should value a reward that can be attained immediately (say, 100 points for acquiring cherries that are only one pixel’s distance away from Pac-Man) more highly than an equivalent reward that would require more timesteps to attain (100 points for cherries that are a distance of 20 pixels away). Immediate reward is more valuable than some distant reward, because we can’t bank on the distant reward: A ghost or some other hazard could get in Pac-Man’s way. 
