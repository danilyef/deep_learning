# Probabilistic Models

Implementation of probabilistic graphical models, inference algorithms, and learning techniques for Bayesian networks.

## Contents

### Inference by Enumeration
| Notebook | Description |
|----------|-------------|
| `inference_by_enumeration.ipynb` | Exact inference through variable elimination and enumeration |

### Approximate Inference
| Notebook | Description |
|----------|-------------|
| `BayesNet Introduction.ipynb` | Introduction to Bayesian network structure and semantics |
| `rejection_sampling.ipynb` | Monte Carlo sampling with rejection for probabilistic queries |
| `likelihood_weighting.ipynb` | Importance sampling for efficient approximate inference |

### Hidden Markov Models
| Notebook | Description |
|----------|-------------|
| `hmm_algorithms.ipynb` | Forward-backward algorithm, Viterbi decoding, and Baum-Welch learning |

### Parameter Learning
| Notebook | Description |
|----------|-------------|
| `BayesNet Introduction.ipynb` | Bayesian network fundamentals and conditional probability tables |
| `Problem 1.ipynb` | Maximum Likelihood Estimation for CPT parameters |
| `Problem 2.ipynb` | Bayesian parameter estimation with prior distributions |

### Structure Learning
| Notebook | Description |
|----------|-------------|
| `structural_learning.ipynb` | Learning network topology from data using scoring functions |

## Key Concepts

- **Bayesian Networks**: Directed acyclic graphs representing conditional dependencies
- **Exact Inference**: Computing posterior probabilities through enumeration
- **Approximate Inference**: Sampling-based methods for intractable distributions
- **HMM**: Temporal models for sequential observations with hidden states
- **Parameter Learning**: Estimating CPT entries from observed data
- **Structure Learning**: Discovering graph topology from data

## Supporting Modules

- `bayesian_network.py` - Core Bayesian network class implementation
- `utils.py` - Helper functions for inference and learning algorithms
