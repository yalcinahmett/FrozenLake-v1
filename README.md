# FrozenLake-v1
This repository contains my implementation of Q-learning for solving the FrozenLake environment. I applied Q-learning, a reinforcement learning algorithm, to teach an agent to navigate a frozen lake and reach the goal. <br>
I used this [article](https://pages.github.com/](https://towardsdatascience.com/q-learning-for-beginners-2837b777741)https://towardsdatascience.com/q-learning-for-beginners-2837b777741).<br>
I have used Spyder IDE with Anaconda for coding. <br>
I faced some errors in the code lines from the article and changed some code lines. <br>
Here is some of them:
> I was facing an error like the image.
> 
> I noticed that state was a tuple.
> I added [0] for returning an integer value which is "0".
```python
for _ in range(episodes):
    state = (environment.reset()[0])
    done = False
```

> There was an error with "new_state, reward, done, info = environment.step(action)".
```python
ValueError: too many values to unpack (expected 4)
```
> I had to replace "new_state, reward, done, info = environment.step(action)" and figured out the error. Then I also changed all 'new_state' with 'obs'.
```python
        obs, reward, done, truncated, info = environment.step(action)
```
Then all the q-learning algorithm worked as I intended. <br>
## So I can summarize how the code works.

> These are the libraries we should use.
```python
import gym
import numpy as np
import matplotlib.pyplot as plt
```

> I initialized the FrozenLake environment, then created the qtable with zeros.
```python
environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode='human')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))
```

> These are the parameters in the q-learning formula.
```python
# Hyperparameters
episodes = 350        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
```

> Output: RESIM EKLE
```python
print('Q-table before training:')
print(qtable)
```

> For teaching the agent.
```python
for _ in range(episodes):
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
```

> Output: RESIM EKLE
```python
print('Q-table after training:')
print(qtable)
```

> Seeing the result in a plot like this: RESIM EKLE
```python
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()
```

> For learning the success rate.
```python
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
```

>For looking at the route that agent takes. When I run this code int the end I can see the route is [1,1,2,2,1,2]
```python
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
```

### I have tried to use this code effectively but I faced some errors. I tried to solve them and now this works as I intented. <br>
## Thank you for reading!
