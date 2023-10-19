import numpy as np
import matplotlib.pyplot as plt


def plot_value_function(V, title="Value Function"):
    min_x = min((k[0] for k in V.keys()))
    max_x = max((k[0] for k in V.keys()))
    min_y = min((k[1] for k in V.keys()))
    max_y = max((k[1] for k in V.keys()))

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    plt.set_cmap('RdBu_r')

    fig, ax = plt.subplots(2, figsize=(8, 8))
    heatmap = ax[0].pcolor(Z_ace, vmin=-1., vmax=1.)
    cbar = fig.colorbar(heatmap, ax=ax[0])

    ax[0].set_xticks(np.arange(Z_ace.shape[1]) + 0.5)
    ax[0].set_yticks(np.arange(Z_ace.shape[0]) + 0.5)

    ax[0].set_xticklabels(x_range)
    ax[0].set_yticklabels(y_range)

    ax[0].set_title(title + ' (useable ace)')
    ax[0].set_ylabel("Dealer showing")
    ax[0].set_xlabel("Player sum")

    heatmap = ax[1].pcolor(Z_noace, vmin=-1., vmax=1.)
    cbar = fig.colorbar(heatmap, ax=ax[1])

    ax[1].set_xticks(np.arange(Z_noace.shape[1]) + 0.5)
    ax[1].set_yticks(np.arange(Z_noace.shape[0]) + 0.5)
    ax[1].set_xticklabels(x_range)
    ax[1].set_yticklabels(y_range)

    ax[1].set_title(title + ' (no ace)')
    ax[1].set_ylabel("Dealer showing")
    ax[1].set_xlabel("Player sum")

    plt.subplots_adjust(hspace=0.3)

    plt.show()


def plot_policy(policy, title="Policy"):
    min_x = min(k[0] for k in policy.keys())
    max_x = max(k[0] for k in policy.keys())
    min_y = min(k[1] for k in policy.keys())
    max_y = max(k[1] for k in policy.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    Z_noace = np.apply_along_axis(lambda _: np.argmax(policy[(_[0], _[1], False)]), 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: np.argmax(policy[(_[0], _[1], True)]), 2, np.dstack([X, Y]))

    fig, ax = plt.subplots(2, figsize=(8, 8))
    heatmap = ax[0].pcolor(Z_ace, cmap='Blues', vmin=0, vmax=1.)

    ax[0].set_xticks(np.arange(Z_ace.shape[1]) + 0.5)
    ax[0].set_yticks(np.arange(Z_ace.shape[0]) + 0.5)

    ax[0].set_xticklabels(x_range)
    ax[0].set_yticklabels(y_range)

    ax[0].set_title(title + ' (useable ace), Blue means get another card')
    ax[0].set_ylabel("Dealer showing")
    ax[0].set_xlabel("Player sum")

    heatmap = ax[1].pcolor(Z_noace, cmap='Blues', vmin=0, vmax=1.)

    ax[1].set_xticks(np.arange(Z_noace.shape[1]) + 0.5)
    ax[1].set_yticks(np.arange(Z_noace.shape[0]) + 0.5)
    ax[1].set_xticklabels(x_range)
    ax[1].set_yticklabels(y_range)

    ax[1].set_title(title + ' (no ace), Blue means get another card')
    ax[1].set_ylabel("Dealer showing")
    ax[1].set_xlabel("Player sum")

    plt.subplots_adjust(hspace=0.3)

    plt.show()
