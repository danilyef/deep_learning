# Reinforcement Learning

Comprehensive collection of reinforcement learning algorithms, from foundational methods to deep RL and imitation learning.

## Contents

### Exploration-Exploitation
| Notebook | Description |
|----------|-------------|
| `k_armed_bandit.ipynb` | Multi-armed bandit algorithms: ε-greedy, UCB, Thompson Sampling |

### Dynamic Programming
| Notebook | Description |
|----------|-------------|
| `dynamic_programming.ipynb` | Policy Iteration and Value Iteration for MDPs with known dynamics |

### Monte Carlo Methods
| Notebook | Description |
|----------|-------------|
| `monte_carlo.ipynb` | First-visit and every-visit MC for prediction and control |

### Temporal Difference Learning
| Notebook | Description |
|----------|-------------|
| `temporal_difference_learning.ipynb` | TD(0), SARSA, Q-Learning, and Expected SARSA algorithms |

### Deep Reinforcement Learning
| Notebook | Description |
|----------|-------------|
| `DQN.ipynb` | Deep Q-Network with experience replay and target networks |
| `policy_gradient.ipynb` | REINFORCE algorithm and variance reduction techniques |
| `PPO.ipynb` | Proximal Policy Optimization with clipped objective |

### Imitation Learning
| Notebook | Description |
|----------|-------------|
| `DAgger.ipynb` | Dataset Aggregation for learning from expert demonstrations |

## Key Concepts

- **Bandits**: Balancing exploration and exploitation under uncertainty
- **Dynamic Programming**: Optimal control with complete environment models
- **Monte Carlo**: Model-free learning from complete episodes
- **Temporal Difference**: Bootstrapping for online, incremental learning
- **DQN**: Function approximation with neural networks for value estimation
- **Policy Gradient**: Direct optimization of parameterized policies
- **PPO**: Stable policy updates with trust region constraints
- **Imitation Learning**: Learning policies from expert behavior

## Environments

- Custom GridWorld environments (FishLake, CliffWalking)
- OpenAI Gym integration
- Blackjack for Monte Carlo methods

## Supporting Modules

- `testing.py` - Evaluation and visualization utilities
- `gym_gridworlds/` - Custom Gym environment implementations
