# Deep Learning

Implementation of fundamental deep learning architectures and techniques using PyTorch and NumPy.

## Contents

### Autoencoders
| Notebook | Description |
|----------|-------------|
| `autoencoder_pytorch.ipynb` | Standard autoencoder implementation for dimensionality reduction and feature learning |
| `vae_pytorch.ipynb` | Variational Autoencoder with reparameterization trick for generative modeling |

### Recurrent Neural Networks
| Notebook | Description |
|----------|-------------|
| `rnn_numpy.ipynb` | Vanilla RNN implementation from scratch using NumPy with backpropagation through time |
| `rtrl_numpy.ipynb` | Real-Time Recurrent Learning algorithm for online training of RNNs |
| `memory_issue_rnn.ipynb` | Analysis of vanishing/exploding gradients in recurrent architectures |

### LSTM
| Notebook | Description |
|----------|-------------|
| `lstm_pytorch.ipynb` | Long Short-Term Memory network for sequence modeling and text generation |

### Dimensionality Reduction
| Notebook | Description |
|----------|-------------|
| `t_sne.ipynb` | t-Distributed Stochastic Neighbor Embedding implementation for visualization |

## Key Concepts

- **Autoencoders**: Unsupervised learning of efficient data encodings
- **VAE**: Probabilistic generative models with latent space regularization
- **RNN/LSTM**: Sequential data processing with temporal dependencies
- **RTRL**: Online learning algorithm for recurrent networks
- **t-SNE**: Non-linear dimensionality reduction preserving local structure

## Frameworks

- PyTorch (autoencoders, LSTM)
- NumPy (RNN implementations from scratch)
