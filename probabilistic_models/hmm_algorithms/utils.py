import numpy as np
import matplotlib.pyplot as plt


class HMM:
    def __init__(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray):
        """
        Datastructure that holds the probability tables for
        a discrete observation HMM having N possible states
        and M possible observations.

        :param pi: Initial probabilities (vector of size N)
        :param A: Transition probabilities (matrix of size NxN)
        :param B: Observation probabilities (matrix of size MxN)
        """
        num_states = len(pi)
        assert pi.shape == (num_states,)
        assert A.shape == (num_states, num_states)
        assert np.allclose(A.sum(axis=1), 1)
        assert B.shape[1] == num_states
        assert np.allclose(B.sum(axis=0), 1)
        
        self.num_states = num_states
        self.pi = pi
        self.A = A
        self.B = B


def get_weather_example(sequence=[2, 1, 0]):
    """
    Returns the weather HMM from the lecture slides.

    :param sequence: the observed sequence
    :return: the HMM and the sequence
    """    
    
    # rainy, cloudy, sunny = 0, 1, 2
    # dry, medium, humid = 0, 1, 2
    
    # transition model
    # fist dimension s_t, second dimension s_{t + 1}
    # P( s_{t + 1} | s_t ) = A[s_{t + 1}, s_t]
    A = np.array([
        [.4, .3 ,.3],
        [.2, .6, .2],
        [.1, .1, .8]
     ])
    
    # observation model
    # fist dimension o, second dimension s
    B = np.array([
        [.1, .3, .6],
        [.5, .4, .2],
        [.4, .3, .2]
    ])
    
    # initial distribution model
    pi = np.array([.2, .3, .5])
    
    return HMM(pi, A, B), np.array(sequence)


def plot_states(path: np.ndarray, values: np.ndarray, title: str=None, state_names: list=['absent', 'sleigh', 'chimney', 'tree']) -> None:
    """
    Plots the results of a algorithm. Black is 1, white is 0.

    :param path: List or array of state ids
    :param values: Detailed values returned by the algorithm (forward-variables, ...)
    :param title: Title of the plot
    :param state_names: Names of the states
    """
    
    f = plt.figure(figsize=(12,4))
    ax = f.add_subplot(111)
    if title is not None:
        ax.set_title(title)
                    
    img = ax.imshow(values.T, interpolation='none', cmap=plt.cm.Greys, vmin=0, vmax=1)
    
    yticks = range(values.shape[1])
    ax.set_yticks(yticks)

    ax.set_yticklabels([state_names[yt] for yt in  yticks])
    
    ax.set_xticks(range(values.shape[0]))
    ax.set_xlabel('Time')
    
    ax.tick_params(top='off', right='off', left='off')
    
    for i in range(len(path)-1):
        ax.arrow(i+0.1, path[i], 0.9, path[i+1] - path[i], length_includes_head=True, head_width=0.25, fc='r', ec='r')
        
    plt.show()