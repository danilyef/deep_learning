# Deep Learning & AI Methods

A curated collection of Jupyter notebooks featuring implementations of foundational and advanced algorithms in deep learning, reinforcement learning, and probabilistic modeling.

## Overview

This repository serves as a comprehensive reference for core AI/ML techniques, with implementations built from first principles using NumPy and PyTorch. Each module includes detailed explanations, mathematical derivations, and practical examples.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| [`deep_learning/`](./deep_learning/) | Neural network architectures including autoencoders, RNNs, LSTMs, and dimensionality reduction |
| [`probabilistic_models/`](./probabilistic_models/) | Bayesian networks, inference algorithms, and parameter/structure learning |
| [`reinforcement_learning/`](./reinforcement_learning/) | RL algorithms from bandits to policy gradients and deep RL |

## Topics Covered

### Deep Learning
- Autoencoders & Variational Autoencoders (VAE)
- Recurrent Neural Networks (RNN) & LSTM
- Real-Time Recurrent Learning (RTRL)
- t-SNE Dimensionality Reduction

### Probabilistic Models
- Bayesian Network Inference
- Hidden Markov Models (HMM)
- Approximate Inference (Rejection Sampling, Likelihood Weighting)
- Parameter & Structure Learning

### Reinforcement Learning
- Multi-Armed Bandits
- Dynamic Programming (Value/Policy Iteration)
- Monte Carlo Methods
- Temporal Difference Learning (SARSA, Q-Learning)
- Deep Q-Networks (DQN)
- Policy Gradient Methods & PPO
- Imitation Learning (DAgger)

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Jupyter Notebook
- Additional dependencies as specified in individual notebooks

## Usage

Clone the repository and navigate to the desired topic:

```bash
git clone https://github.com/username/deep_learning.git
cd deep_learning
jupyter notebook
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
