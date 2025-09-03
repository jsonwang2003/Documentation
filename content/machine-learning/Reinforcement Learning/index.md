# Reinforcement Learning

> Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards. The goal is to learn a policy that maximizes cumulative reward over time.

## Overview

Unlike supervised learning, RL does not rely on labeled input-output pairs. Instead, it learns from experience, using trial and error to discover optimal behaviors. RL is particularly suited for sequential decision-making problems where actions influence future states and rewards.

RL problems are typically formalized as **Markov Decision Processes (MDPs)**, defined by:

- **States (S)**: Represent the environment at a given time  
- **Actions (A)**: Choices available to the agent  
- **Transition Function (T)**: Probability of moving from one state to another  
- **Reward Function (R)**: Feedback signal for each action taken  
- **Policy (π)**: Strategy that maps states to actions

## Key Concepts

- **Exploration vs. Exploitation**: Balancing the search for new strategies with leveraging known ones  
- **Value Function**: Estimates expected future rewards from a state or state-action pair  
- **Policy Optimization**: Learning the best strategy for decision-making  
- **Temporal Difference Learning**: Updates value estimates based on differences between successive predictions  
- **Model-Free vs. Model-Based RL**: Whether the agent learns without or with an explicit model of the environment  
- **On-Policy vs. Off-Policy**: Whether the agent learns from its own actions or from a separate behavior policy

## Related Topics

- [[Monte Carlo Tree Search]] — Monte Carlo Tree Search: A planning algorithm often used in hybrid RL systems for decision-making under uncertainty  
- [[Policy Gradient Methods]] — Probabilistic Graphical Models: Useful for modeling structured uncertainty and reasoning in model-based or hybrid RL approaches

## Applications

- Autonomous control and navigation  
- Game-playing agents  
- Robotics and manipulation  
- Industrial process optimization  
- Financial decision-making  
- Multi-agent coordination

## Suggested Links

- [[Reinforced Deep Learning Models/index|Reinforced Deep Learning Models]] — Deep RL algorithms using neural networks  
- [[Hybrid Deep Learning Models]] — For architectures combining RL with other paradigms  
- [[Model Evaluation]] — For metrics and validation strategies in RL  
- [[Supervised Learning/index|Supervised Learning]] — For comparison with supervised paradigms
