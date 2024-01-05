# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:49:02 2024

@author: yahme
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode='human')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# We re-initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hyperparameters
episodes = 350        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor

# List of outcomes to plot
outcomes = []

print('Q-table before training:')
print(qtable)

# Training
for _ in range(episodes):
    # I added [0] for solving the error
    state = (environment.reset()[0])
    done = False
    
    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        ## I changed the state tuple to state[0] and it doesn't give errors for now.
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])

        # If there's no best action (only zeros), take a random one
        else:
          action = environment.action_space.sample()
             
        # Implement this action and move the agent in the desired direction
        # I had to change "new_state, reward, done, info = environment.step(action)" and figured out the error
        obs, reward, done, truncated, info = environment.step(action)
        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[obs]) - qtable[state, action])
        
        # Update our current state
        state = obs

        # If we have a reward, it means that our outcome is a success
        if reward:
          outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()

# for learning the success rate

episodes = 100
nb_success = 0

# Evaluation
for _ in range(episodes):
    state = environment.reset()[0]
    done = False
    
    # Until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])

        # If there's no best action (only zeros), take a random one
        else:
          action = environment.action_space.sample()
             
        # Implement this action and move the agent in the desired direction
        obs, reward, done, truncated, info = environment.step(action)

        # Update our current state
        state = obs

        # When we get a reward, it means we solved the game
        nb_success += reward

# Let's check our success rate!
print (f"Success rate = {nb_success/episodes*100}%")

# for looking at the route that agent takes
from IPython.display import clear_output
import time 

state = environment.reset()[0]
done = False
sequence = []

while not done:
    # Choose the action with the highest value in the current state
    if np.max(qtable[state]) > 0:
      action = np.argmax(qtable[state])

    # If there's no best action (only zeros), take a random one
    else:
      action = environment.action_space.sample()
    
    # Add the action to the sequence
    sequence.append(action)

    # Implement this action and move the agent in the desired direction
    obs, reward, done, truncated, info = environment.step(action)

    # Update our current state
    state = obs

    # Update the render
    clear_output(wait=True)
    environment.render()
    time.sleep(1)

# If I wanna see the route I should run "sequence"
# Then I can see the route is [1,1,2,2,1,2]
