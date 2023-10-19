import numpy as np
from bayesian_network import BayesNet, Variable
import matplotlib.pyplot as plt
import networkx as nx
    
def compare_bayes_net(bn1: BayesNet, bn2: BayesNet) -> None:
    
    vars1 = [(i[0], i[1:], pdt) for i, pdt in zip(bn1.indices, bn1.pdts)]
    vars2 = [(i[0], i[1:], pdt) for i, pdt in zip(bn2.indices, bn2.pdts)]
    
    vars1.sort(key=lambda x: x[0])
    vars2.sort(key=lambda x: x[0])
    
    assert_true = True
    
    if not len(vars1) == len(vars2):
        print(f'Expected {len(vars1)} variables, got {len(vars2)}.')
        assert_true = False
        
    
    for v1, v2 in zip(vars1, vars2):
        
        if not v1[0] == v2[0]:
            print(f'Expected variable with ID: {v1[0]}, actual: {v2[0]}.')
            assert_true = False
            
        for p1 in v1[1]:
            if not p1 in v2[1]:
                print(f'Variable {v1[0]}: Expected parents: {v1[1]}, actual parents: {v2[1]}.')
                assert_true = False
                
        for p2 in v2[1]:
            if not p2 in v1[1]:
                print(f'Variable {v1[0]}: Expected parents: {v1[1]}, actual parents: {v2[1]}.')
                assert_true = False
        
        pdt1 = v1[2].transpose((0,) + tuple(np.argsort(v1[1])+1))
        pdt2 = v2[2].transpose((0,) + tuple(np.argsort(v2[1])+1))
        if not np.all(np.isclose(pdt1, pdt2, atol=0.001)):
            print(f'PDTs of {v1[0]} and {v2[0]} do not match')
            assert_true = False

    assert assert_true, 'See error message above.'
        

def get_default_bayes_net(resampling_distribution=None):
    _A_, _B_, _C_, _D_, _E_ = 0, 1, 2, 3, 4

    A = np.array([0.2, 0.8])
    B_A = np.array([[0.9, 0.2], [0.1, 0.8]])
    C = np.array([0.9, 0.1])
    D_BC = np.array([[[0.1, 0.2], [0.99, 0.8]], [[0.9, 0.8], [0.01, 0.2]]])
    E_C = np.array([[0.7, 0.4], [0.3, 0.6]])

    return BayesNet((A, [_A_]), (B_A, [_B_, _A_]), (C, [_C_]), (D_BC, [_D_, _B_, _C_]), (E_C, [_E_, _C_]), resampling_distribution=resampling_distribution)



def maximum_likelihood_estimate(data: np.ndarray, variable_id: int, parent_ids: tuple=tuple(), laplace=1) -> np.ndarray:
    """
    Estimates the conditional probability distribution of a (discrete) variable from data.
    :param data: data to estimate distribution from
    :param variable_id: column index corresponding to the variable we estimate the distribution for
    :param parent_ids: column indices of the variables the distribution is conditioned on
    :param laplace: Laplace smoothing with this alpha
    :returns: estimated conditional probability distribution table
    """
    assert type(parent_ids) == tuple
    
    # mapping of axis to variable_id,
    # e.g. the variable with id variable_ids[i] is on axis i of the CPDT
    variable_ids = (variable_id,) + parent_ids
    
    # create empty CPDT
    cpdt = np.zeros((2,)*len(variable_ids))

    # count
    for datum in data:
        index = tuple([datum[v] for v in variable_ids])
        cpdt[index] += 1
    
    cpdt += laplace
    
    # normalize
    cpdt = cpdt / cpdt.sum(axis=0, keepdims=True)
    
    return cpdt

def log_likelihood(data: np.ndarray, bayes_net: BayesNet) -> float:
    """
    Computes the log-likelihood of a given Bayesian network relative to the data.
    :param data: data to compute the log-likelihood relative to.
    :param bayes_net: Bayesian network model.
    :returns: the log-likelihood of the Bayesian network relative to the data.
    """    

    ll = 0
    
    def log_prob(datum: np.ndarray):
        p = 0
        for variable in bayes_net:
            p += np.log(variable(datum)[datum[variable.id]])
        return p
    
    for sample in data:
        ll += log_prob(sample)   
    
    return ll



def draw_graph(bayes_net: BayesNet, node_names: list = None, pos: dict = None) -> None:
    """
    Draws the Bayesian net.
    :param bayes_net: a BayesNet object representing the graph structure to draw.
    :param node_names: Display Name of the variables. 
                       Defaults to values for the stroke-example.
    :param pos: position of the variables (dict: key=variable_name, value: list: x, y). 
                Defaults to values for the stroke-example.
    """
    
    if node_names == None or pos == None:
        node_names = ['Alc', 'High BP', 'Str.', 'Conf.', 'Vert.']
        pos = {'Alc': [0., 3.],
                   'High BP': [.5, 3.],
                   'Str.': [.75, 2.5],
                   'Conf.': [0., 2.],
                   'Vert.': [.5, 2.]
            }

    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    for i in bayes_net.indices:
        for p in i[1:]:
            G.add_edge(node_names[p], node_names[i[0]])       

    nx.draw(G, pos=pos, with_labels=True, node_size=3000, node_color='#F5F5F5')
    plt.show()

    
def evaluate_bayes_net(bayes_net: BayesNet, train_set: np.ndarray, test_set: np.ndarray) -> tuple:
    """
    Computes the mean likelihood of the Bayesian network under the training and test data
    :param bayes_net: a BayesNet object, representing the model.
    :param train_set: the training set, a NumPy array of shape (num_samples, len(bayes_net)).
    :param test_set: the test set, a Numpy array (num_samples, len(bayes_net)).
    :return: a tuple: (log_likelihood_of_training_data, log_likelihood_of_test_data)
    """
    train_logl = log_likelihood(train_set, bayes_net) / len(train_set)
    test_logl = log_likelihood(test_set, bayes_net) / len(test_set)
    return train_logl, test_logl


def compare_train_size(models: dict, data_sets: np.ndarray, test_set: np.ndarray):
    """
    Plots the mean log-likelihood of the models relative to the test data.
    """
    given_test_loss = [evaluate_bayes_net(models[data_set_id]['pre-defined'], data_set, test_set)[1] for data_set_id, data_set in zip(['small', 'medium', 'big'], data_sets)] 
    ord_test_loss = [evaluate_bayes_net(models[data_set_id]['ordered'], data_set, test_set)[1] for data_set_id, data_set in zip(['small', 'medium', 'big'], data_sets)] 
    rev_test_loss = [evaluate_bayes_net(models[data_set_id]['reverse'], data_set, test_set)[1] for data_set_id, data_set in zip(['small', 'medium', 'big'], data_sets)]  
    rnd_test_loss = [evaluate_bayes_net(models[data_set_id]['random'], data_set, test_set)[1] for data_set_id, data_set in zip(['small', 'medium', 'big'], data_sets)] 
    ord_ll_test_loss = [evaluate_bayes_net(models[data_set_id]['log-likelihood'], data_set, test_set)[1] for data_set_id, data_set in zip(['small', 'medium', 'big'], data_sets)] 
    
    plt.figure(figsize=(15, 10))
    plt.plot(ord_test_loss, marker='d', label='ordered')
    plt.plot(rev_test_loss, marker='d', label='reverse')
    plt.plot(rnd_test_loss, marker='d', label='random')
    plt.plot(ord_ll_test_loss, marker='d', label='ordered log-l')
    plt.plot(given_test_loss, marker='d', label='given structure')
    plt.xlim(-0.25, 2.25)
    plt.xticks([0, 1, 2], ['Small', 'Medium', 'Large'])
    plt.ylabel('Test Data Log-Likelihood')
    plt.xlabel('Train Set')
    plt.legend(loc='best')
    plt.show()
    
def plot_compare_orders(models, data_sets, test_set):
    """
    Plots the mean log-likelihood of the models relative to the test and traing data.
    """
    for data_set_id, data_set in zip(['small', 'medium', 'big'], data_sets):
        train_losses = []
        test_losses = []
        for combination_id in [ 'pre-defined', 'ordered', 'reverse', 'random', 'log-likelihood']:
            tr_ll, te_ll = evaluate_bayes_net(models[data_set_id][combination_id], data_set, test_set)
            train_losses.append(tr_ll)
            test_losses.append(te_ll)
            
        f = plt.figure(figsize=(15, 5))
        f.suptitle(f'Dataset: {data_set_id}', fontsize=15)
        ax = f.add_subplot(121)
        ax.plot(train_losses, marker='d', linestyle='--')
        ax.set_xlim(-0.25, len(train_losses) - 0.75)
        ax.set_title('Log-Likelihood train data')
        ax.set_xticks(range(len(train_losses)))
        ax.set_xticklabels(['Pre-def', 'Ord', 'Rev', 'Rnd', 'Ord-LL'])
        ax = f.add_subplot(122)
        ax.plot(test_losses, marker='d', color='g', linestyle='--')
        ax.set_xlim(-0.25, len(train_losses) - 0.75)
        ax.set_title('Log-Likelihood test data')
        ax.set_xticks(range(len(train_losses)))
        ax.set_xticklabels(['Pre-def', 'Ord', 'Rev', 'Rnd', 'Ord-LL'])
        plt.show()
        
        
def sample_categorical(dist: np.ndarray) -> np.int64:
    """
    Draws a single sample from a categorical distribution.
    :param dist: NumPy array listing the probability of each outcome.
    :returns: Index of the sampled element.
    """

    assert type(dist) == np.ndarray
    assert dist.ndim == 1
    assert np.isclose(dist.sum(), 1)

    return np.random.choice(len(dist), p=dist)