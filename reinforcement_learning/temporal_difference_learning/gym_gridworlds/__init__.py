import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register


register(
    id='Cliff-v0',
    entry_point='gym_gridworlds.envs:CliffEnv',
)


register(
    id='QBiasTest-v0',
    entry_point='gym_gridworlds.envs:QBiasTestEnv',
)


def random_policy(env):
    nS = env.observation_space.n
    nA = env.action_space.n
    return np.ones((nS, nA)) / nA


# this is a helper function that turns a V represented as a dict into an array
def dict_to_array(env, d):
    V = np.zeros(env.observation_space.n)
    for state, value in d.items():
        V[state] = value
    return V


def print_value_function(env, V_dict):
    V = dict_to_array(env, V_dict)
    # for frozenlake, we reshape this array to reflect
    # its 'spatial' properties - so the state value function
    # has the same shape as the lake
    print(V.reshape(env.nrow, env.ncol))


def plot_policy(env, policy):
    fig, ax = plt.subplots()
    xs = np.arange(env.ncol)
    ys = np.arange(env.nrow)
    xx, yy = np.meshgrid(xs, ys)

    direction = [
        np.array((-1, 0)),  # left
        np.array((0, -1)),  # down
        np.array((+1, 0)),  # right
        np.array((0, +1)),  # up
    ]

    # we need a quiver for each of the four action
    quivers = list()
    for a in range(env.action_space.n):
        quivers.append(list())

    # we parse the textual description of the lake
    desc = env.desc.reshape(env.observation_space.n)
    for s in range(env.observation_space.n):
        if desc[s] in (b'H', b'G'):
            for a in range(4):
                quivers[a].append((0., 0.))
        else:
            for a in range(4):
                quivers[a].append(direction[a] * policy[s][a])

    # plot each quiver
    for quiver in quivers:
        q = np.array(quiver)
        ax.quiver(xx, yy, q[:, 0], q[:, 1], units='xy', scale=1.5)

    # set axis limits / ticks / etc... so we have a nice grid overlay
    ax.set_xlim((xs.min() - 0.5, xs.max() + 0.5))
    ax.set_ylim((ys.max() + 0.5, ys.min() - 0.5))  # y axis is flipped, so it corresponds to what imshow shows

    ax.set_xticks(xs)
    ax.set_yticks(ys)

    # major ticks
    sx = env.ncol
    sy = env.nrow
    ax.set_xticks(np.arange(0, sx, 1))
    ax.set_yticks(np.arange(0, sy, 1))

    # minor ticks
    ax.set_xticks(np.arange(-.5, sx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, sy, 1), minor=True)

    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)

    ax.set_aspect(1)
    plt.show()


def plot_value_function(env, V_dict, vmin=None, vmax=None):
    """
    This is going to show the value function 'V' as an image
    """
    V = dict_to_array(env, V_dict)
    if vmin is None:
        vmin = V.min()
    if vmax is None:
        vmax = V.max()

    mask = np.full_like(V, False, dtype=np.bool)
    # mark the terminal states as having 'no value'
    # to do that, we parse the textual description of the gridworld
    ux = []
    uy = []
    desc = env.desc.reshape(env.observation_space.n)
    for s in range(env.observation_space.n):
        if desc[s] in (b'H', b'G'):
            y, x = divmod(s, env.ncol)
            ux.append(x)
            uy.append(y)
            mask[s] = True

    # mask out all terminal states
    masked = np.ma.masked_where(mask, V)

    fig, ax = plt.subplots()
    im = ax.imshow(
        masked.reshape((env.nrow, env.ncol)),
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        aspect=1
    )

    ax.scatter(ux, uy, marker='x', color='r')

    xs = np.arange(env.ncol)
    ys = np.arange(env.nrow)

    # set axis limits / ticks / etc... so we have a nice grid overlay
    ax.set_xlim((xs.min() - 0.5, xs.max() + 0.5))
    ax.set_ylim((ys.max() + 0.5, ys.min() - 0.5))  # y axis is flipped, so it corresponds to what imshow shows

    ax.set_xticks(xs)
    ax.set_yticks(ys)

    # major ticks
    sx = env.ncol
    sy = env.nrow
    ax.set_xticks(np.arange(0, sx, 1))
    ax.set_yticks(np.arange(0, sy, 1))

    # minor ticks
    ax.set_xticks(np.arange(-.5, sx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, sy, 1), minor=True)

    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)

    plt.colorbar(im)
    plt.show()
