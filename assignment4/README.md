# COMP4901J
This is the assignment code for COMP4901J.

## Assignment #4

### Q-learning, Deep Q-Networks, Policy Gradient, Model Network

In this assignment you will practice writing simple reinforcement learning algorithms, and training DQN and Policy Gradient to solve the classical CartPole and WorldNavigate problems. The goals of this assignment are as follows:

- Practice how to use the OpenAI Gym interfaces
- Understand and be able to implement Q-table and Q-network learning algorithms
- Understand convolutional layers and how they can be applied in Q-learning and policy gradient methods
- Implement DQN to solve the world navigation task
- Implement two improved DQNs, the Double Q-learning and Duel DQN
- Build a policy-gradient based agent that can solve the CartPole task
- Combine model network and policy network without actually training on real environment
Get the code as a zip file here. [Updated: 16 Nov 2017]
Submitting your work:

Whether you work on the assignment locally or in the labs, once you are doneworking run the Python code collectSubmission.py; this will produce a file called assignment4.7z. Upload this file under CASS. You can find CASS instruction of uploading your submission here.

### Q1: Basic Q-learning algorithms (10 points)

The Jupyter notebook Q_learning_Basic.ipynb will walk you through the implementation of two vanilla Q-learning algorithms, the Q-table and Q-network, with FrozenLake task as example.
### Q2: World navigation with DQN (30 points)

The Jupyter notebook DQN_WorldNavigate.ipynb will walk you through the implementation of Deep Q-network. You will also implement two simple additional improvements to the DQN architecture, Double DQN and Dueling DQN, that allow for improved performance.
### Q3: CartPole with Policy Gradient (30 points)

The Jupyter notebook PG_CartPole.ipynb will walk you through the implementation of a policy-gradient based agent that can solve the classic CartPole task.
### Q4: Model-based Reinforcement Learning (30 points)

The Jupyter notebook Model_Policy_Network.ipynb will introduce you how to combine model network and policy network so that there is no need to actually train on real environment.
### Q5: Do something extra! (up to +xx points)

Since we have two networks involved in Q4, there are plenty of hyper-parameters to adjust in order to improve performance or efficiency. You are encouraged to play with them in order to discover better means of combining the models and get better results.
### Q6: Do even more! (up to +xx points)

With Q1-Q5 we have several fully-functional reinforcement learning agents. Our agents are still far from the state of the art though. Try to make some improvements (e.g. by using more complex networks) of the current models, and try getting your agent to perform well in one of the ATARI games (e.g. Pong, LunarLander...). To score full credits you must include a clear write-up possibly with clear drawings.