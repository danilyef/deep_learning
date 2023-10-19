import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAP = [
    "FFG",
    "FFF",
    "SFF"
]


class QBiasTestEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = desc = np.asarray(MAP, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-12, 10)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: list() for a in range(nA)} for s in range(nS)}

        self.rewards = []

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]

            # this is a tiny bit different from the description in the
            # paper. in the paper, it says "in the goal state every action
            # yields +5 and ends an episode. the optimal policy ends an
            # episode after five actions.

            # we modify this to stop as soon as we transition to the 'G' state
            done = False
            if newletter == b'G':
                reward = 5
                done = True
                yield 1., newstate, reward, done
            else:
                # "... each non-terminating step, the agent receives a random
                # reward of -12 or +10 with equal probability."
                for reward in [-12, 10]:
                    yield 0.5, newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        # if terminal, stay in the same state with p==1
                        # and zero(!) reward
                        # important for DP methods
                        li.append((1., s, 0, True))
                    else:
                        # no non-determinism in 'the cliff' env
                        for entry in update_probability_matrix(row, col, a):
                            li.append(entry)

        super(QBiasTestEnv, self).__init__(nS, nA, P, isd)

    def step(self, action):
        s, r, d, info = super().step(action)
        self.rewards.append(r)
        return s, r, d, info

    def get_rewards(self):
        return self.rewards

    def reset_rewards(self):
        self.rewards = []

    def get_start(self):
        return np.nonzero(np.array(self.desc == b'S').ravel())[0][0]

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


# def main():
#     from pprint import pprint
#     env = QBiasTestEnv()

#     for entry in env.P.items():
#         pprint(entry)

#     print(env.get_start())


# if __name__ == '__main__':
#     main()
