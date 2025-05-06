# This script uses the framework provided by Bernd Schrooten in Deep Reinforcement Learning with Gymnasium published by Packt Publishing
# Code generation assistant: DataCamp AI's Python code generation tool
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gymnasium as gym 

#for i in gym.envs.registry.keys():
#    print(i)

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.n)
# print(env.action_space.sample())

observation, info = env.reset()
#print(observation)
#print(info)

# Create a neural network policy
input_dim = env.observation_space.shape[0]
hidden_dim = 64
output_dim = env.action_space.n

#print(input_dim, output_dim)

def create_policy():
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim).double(),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim).double(),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim).double(),
        nn.Softmax(dim=-1),
    )

policy = create_policy()
observation, _ = env.reset()
tensor = policy(torch.tensor(observation, dtype=torch.float64))

print(tensor)

action = env.action_space.sample(probability=tensor.detach().cpu().numpy())
print(action)

step = env.step(action)
print(step)

# Train and validate policy
# Hyperparameters
learning_rate = 0.01
gamma = 0.99 # Discount factor
epochs = 500

# Optimizer
policy = create_policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Training loop
for episode in range (epochs):
    observation, _ = env.reset()
    episode_rewards = []
    log_probs = []

    while True:
        # Get action probabilities from the policy
        action_probs = policy(torch.tensor(observation, dtype=torch.float64))
        
        # Sample an action from the action probabilities
        #action = torch.multinomial(action_probs, num_samples=1).item()
        action = env.action_space.sample(probability=action_probs.detach().cpu().numpy())

        # Log the probabilities of the action taken
        log_prob = torch.log(action_probs[action]).flatten()
        log_probs.append(log_prob)

        # Take the action in the environment
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_rewards.append(reward)

        observation = next_observation

        if terminated or truncated:
            break

    # Calculate the discounted rewards  
    discounted_rewards = []
    cumulative_reward = 0

    for reward in reversed(episode_rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)

    # Normalize the discounted rewards
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float64)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())

    # Calculate the policy gradient loss
    # policy_loss = []
    # for log_prob, reward in zip(log_probs, discounted_rewards):
    #     policy_loss.append(-log_prob * reward)
    # policy_loss = torch.cat(policy_loss).sum()
    policy_loss = -(torch.cat(log_probs) * discounted_rewards).sum()

    # Backpropagation
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if episode % 20 == 0:
        # Print the total reward for the episode
        print(f"Episode {episode + 1}/{epochs}, Total Reward: {sum(episode_rewards)}")


# Test the trained policy
policy.eval()