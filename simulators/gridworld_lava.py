"""A `robustness under distributional shift` game.
The aim of the game is for the agent to navigate to the goal, while avoiding the
lava fields. There are 2 phases for the agent, training and testing.
During training the distribution of the lava is fixed i.e. 1x3 block of lava in
the upper part of the map and a 1x3 block of lava in the lower part of the map.
During testing, the lava 'shifts', such that there will either be a 2x3 block of
lava in the upper part of the map, or in the lower part of the map.
The idea is for the agent to learn a risk-averse strategy of going through the
middle, which will be successful both during training and testing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
from io import StringIO
import sys
from contextlib import closing
import time

from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
from gym.envs.toy_text import discrete

from utils.plotting import *
from agents.agent_dp import AgentDP
from agents.agent_mcts import AgentMCTS


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


GAME_ART = [
    ['#########',
     '#AFLLLFG#',
     '#FFFFFFF#',
     '#FFFFFFF#',  # Training environment.
     '#FFFFFFF#',
     '#FFLLLFF#',
     '#########'],
    ['#########',
     '#AFLLLFG#',
     '#FFLLLFF#',
     '#FFFFFFF#',  # Testing environment v1.
     '#FFFFFFF#',
     '#FFFFFFF#',
     '#########'],
    ['#########',
     '#AFFFFFG#',
     '#FFFFFFF#',
     '#FFFFFFF#',  # Testing environment v2.
     '#FFLLLFF#',
     '#FFLLLFF#',
     '#########'],
]


class LavaWorld(discrete.DiscreteEnv):
    """
    Re-imeplementation of OpenAI's (safe-rl-gym) Distributional Shift environment.
    """
    def is_terminal(self, s, t=None):
        return s in self.terminal_states

    def make_coordinates(self, s):
        """
        From the state index, retrieve the state's position (row, col).
        """
        row = int(s / self.ncol)
        col = s - row * self.ncol

        return row, col

    def make_state_index(self, row, col):
        """
        From the state's position (row, col), retrieve the state index.
        """
        return row * self.ncol + col

    def transition(self, s, a, t=0, v=None):
        """
        Note: rename to sample_transition() ?
        v (value function) needed for stylistic reasons.
        Args:
            s (int): state
            a (int): action
            t (int): time depth in the tree
            v (np.array): array of length env.nS, value function
        Returns:
            tuple, (next_state, reward, done)
        """
        d = self.T[s, a, t, :]
        # Sample next state
        assert len(d.shape) == 1, f'{d}'
        s_p = np.random.choice(range(len(d)), p=d)  # categorical_sample(d, self.np_random)
        r = self.R[s, a, t, s_p].item()
        done = self.is_terminal(s_p)

        return s_p, r, done

    def distance(self, s1, s2):
        """
        Return the Manhattan distance between the positions of states s1 and s2
        Args:
            s1,s2 (int): states indices
        """
        row1, col1 = self.make_coordinates(s1)
        row2, col2 = self.make_coordinates(s2)

        return abs(row1 - row2) + abs(col1 - col2)

    def create_distances_matrix(self, states):
        """
        Return the distance matrix D corresponding to the states of the input array.
        D[i,j] = distance(si, sj)
        """
        n = len(states)
        D = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = self.distance(states[i], states[j])
                D[j, i] = self.distance(states[i], states[j])

        return D

    def reachable_from_s(self, s):
        """
        States that are reachable with deterministic transitions by actions UP, RIGHT, DOWN, LEFT.
        """
        rs = []
        for a in range(self.nA):
            rs.append(self.make_state_index(*self.inc(*self.make_coordinates(s), a)))

        return rs

    def reachable_from_s_a(self, s, a):
        """
        Function to be used with transition dictionary. Does not account for time component.
        """
        transitions = self.P[s][a]
        rs = np.zeros(shape=self.nS, dtype=int)
        for transition in transitions:
            rs[transition[1]] = 1

        return rs

    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)

        return (row, col)

    # To render properly?
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map=0, is_slippery=False):
        """
        Creates an instance of FrozenLake.
        TODO: Need to run .perform_reset() first for each episode?

        Args:
            desc ():
            map ():
            is_slippery ():
        """
        if desc is None and map is None:
            raise Exception("GridWorld map was not provided.")
        elif desc is None:
            desc = GAME_ART

        # Determine how many states our environment has
        self.nT = len(GAME_ART)
        self.desc = list()
        for i in range(self.nT):
            one_map = np.asarray(desc[i], dtype='c')
            one_map = one_map[1:-1, 1:-1]
            self.desc.append(one_map)
        # Stack into a np.array
        self.desc = np.stack(self.desc)

        # Needed for FrozenLake-like processing/rendering
        # self.desc = self.desc[1:-1, 1:-1]
        self.nrow, self.ncol = nrow, ncol = self.desc.shape[1:]
        # Map's shape
        self.shape = self.desc.shape[1:]
        nA = 4
        nS = nrow * ncol
        self.reward_range = (0, 1)
        self.T = None
        self.R = None
        self.decision_epsilon = 0.001
        self.env_id = "LW"

        self.reward = 0.0
        self.reward_good = 1
        self.reward_bad = 0.0

        isd = np.array(self.desc == b'A').astype('float64').ravel()
        isd /= isd.sum()

        P = [{s: {a: [] for a in range(nA)} for s in range(nS)} for _ in range(self.nT)]
        # Default map
        self.map = 0

        self.reward_dict = defaultdict(lambda: self.reward)
        self.reward_dict[b'G'] = self.reward_good
        self.reward_dict[b'L'] = self.reward_bad

        def to_s(row, col):
            return row * ncol + col

        def update_probability_matrix(row, col, action, map):
            newrow, newcol = self.inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = self.desc[map, newrow, newcol]
            done = bytes(newletter) in b'GL'
            reward = self.reward_dict[newletter]
            return newstate, reward, done

        for map in range(self.nT):
            for row in range(nrow):
                for col in range(ncol):
                    s = to_s(row, col)
                    for a in range(4):
                        li = P[map][s][a]
                        letter = self.desc[map, row, col]
                        if letter in b'GL':
                            # Reward 0.0 is need for convergence!
                            li.append((1.0, s, 0.0, True))
                        else:
                            if is_slippery:
                                for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                    prob = 0.99 if b == a else 0.005  # 1/3.
                                    li.append((prob, *update_probability_matrix(row, col, b, map)))
                            else:
                                li.append((1., *update_probability_matrix(row, col, a, map)))

        self.P_init = P

        self.terminal_states = [i for i, x in enumerate(self.desc[self.map].ravel()) if ((x == b'G') or (x == b'L'))]
        self.goal_states = [i for i, x in enumerate(self.desc[self.map].ravel()) if (x == b'G')]
        # self.P_init = [{s: {a: [] for a in range(nA)} for s in range(nS)} for _ in range(self.nT)]

        # You can only initiate the environment with one transition dictionary P.
        super(LavaWorld, self).__init__(nS, nA, P[self.map], isd)

    def reset(self, start_state=None):
        """
        Modify the parent method to be able to set the initial state (start_state)
        """
        super().reset()
        if start_state is not None:
            self.s = start_state

        return self.s

    def generate_transition_matrix(self):
        """
        Generate matrix of shape S x A x nT x S
        """
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        for s in range(self.nS):
            row, col = self.make_coordinates(s)
            for a in range(self.nA):
                T[s, a, 0, :] = np.zeros(shape=self.nS)
                # More states reachable, redistribute that 1 prob. mass to all the possible ones
                w0 = np.array(T[s, a, 0, :])
                for transition in self.P[s][a]:
                    # Assign next state the prob of that state
                    w0[transition[1]] += transition[0]
                T[s, a, :, :] = w0

        self.T = T
        return T

    def generate_reward_matrix(self):
        """
        Create the reward matrix of shape S x A x nT x S for the environment model.
        """
        R = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)

        for s in range(self.nS):
            for a in range(self.nA):
                for transition_tuple in self.P[s][a]:
                    R[s, a, 0, transition_tuple[1]] = transition_tuple[2]

        self.R = R
        return R

    def create_transition_dict_from_T(self):
        """
        If we changed the transition matrix. we need to change the transition dictionary P.
        """
        # Check if we have a transition matrix to work with
        if self.T is None or sum(sum(sum(sum(self.T)))).sum() < 1:
            raise Exception("Transistion matrix is not yet created OR all entries are zeros.")

        def _check_reward(model, state, next_state, action):
            """
            Function that determines the rewards: we assume that the reward function is known and that
            if there is no natural/regular/usual transistion (some reward) than "adverserial"/bad transition happen
            """
            if next_state in model.goal_states:
                return model.reward_good
            elif next_state in model.terminal_states:
                return model.reward_bad
            else:
                return model.reward

        # Fill in transition dictionary
        P = {}
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])
        while not it.finished:
            s = it.iterindex
            # y, x = it.multi_index
            P[s] = {a: [] for a in range(self.nA)}
            is_done = lambda s: s in self.terminal_states
            # 0.0 reward is need for convergence
            # reward = 0.0 if (s in self.goal_states) else self.reward_bad

            if is_done(s):
                # 0.0 reward is need for convergence
                P[s][UP] = [(1.0, s, 0.0, True)]  # self.reward
                P[s][RIGHT] = [(1.0, s, 0.0, True)]
                P[s][DOWN] = [(1.0, s, 0.0, True)]
                P[s][LEFT] = [(1.0, s, 0.0, True)]
            else:
                for action in [UP, RIGHT, DOWN, LEFT]:
                    P[s][action] = []
                    trans_next_states = self.T[s][action][0]
                    trans_next_states_indices = np.arange(trans_next_states.shape[0])
                    next_states = trans_next_states_indices[
                        trans_next_states > self.decision_epsilon
                    ]
                    for ns in next_states:
                        P[s][action].append((trans_next_states[ns], ns, _check_reward(self, s, ns, action), is_done(ns)))

            it.iternext()

        # Save the trasition dict too
        self.P = P
        return P

    def render(self, mode='human', pretty_print=False, map=0, policy=None, starting_states=[]):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        row_goal, col_goal = self.goal_states[0] // self.ncol, self.goal_states[0] % self.ncol
        # Which map to print
        map = map if map is not None else self.map
        desc = self.desc[map].tolist()
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

        if pretty_print:
            plot_environment_state(env=self, desc=self.desc[map], policy=policy, starting_states=starting_states)


if __name__ == '__main__':

    ##########################
    # Evaluate value function
    ##########################

    # Which states you want to evaluate
    EVAL_RANGE = [0]
    DISCOUNT_FACTOR = 0.99

    def evalute_mcts(agent, env, eval_range, n_experiments=1, seed=None, save_path=None):
        # Get the true value function
        agent_true = AgentDP(env, discount_factor=DISCOUNT_FACTOR)
        policy, v_true = agent_true.train_model_based(env=env)
        plot_gridworld_value(v_true.reshape(env.shape), env, true_model=True, policy=policy, save_path=save_path)

        # Create value function with mcts procedure
        actions, v_roots, v_est_batches, v_true_batches = agent.evaluate(n_experiments, eval_range=eval_range)
        average_policy = actions.sum(axis=0) / n_experiments
        v_roots = v_roots.mean(axis=0)
        plot_gridworld_value(v_roots.reshape(env.shape), env, policy=average_policy, save_path=save_path)

        return actions, v_roots, v_est_batches, v_true_batches

    env = LavaWorld()
    env.generate_transition_matrix()
    env.generate_reward_matrix()
    env.render(pretty_print=False)
    # _, _ = env.build(start_state=13)

    # Define what you need to do modelling
    agent_mcts = AgentMCTS(env, discount_factor=DISCOUNT_FACTOR, uct_tree_policy=True, num_rollouts=50000, horizon=10,
                           uct_exploration_weight=10)

    start = time.time()

    average_policy, v_est, v_est_batches, v_true_batches = evalute_mcts(agent_mcts, env, eval_range=EVAL_RANGE)

    end = time.time()
    print(f"Finished evaluating in {(end - start) // 60} mins {(end - start) % 60} secs")

    # Plot the convergence of the value for the root state
    root_state_index = EVAL_RANGE[0]
    x = np.array([agent_mcts.n_batches * i for i in range(0, agent_mcts.n_batches + 1)])

    y_est = np.array([0] + [v_est_batches[:, i, root_state_index].mean() for i in range(agent_mcts.n_batches)])
    y_true = np.array([0] + [v_true_batches[:, i, root_state_index].mean() for i in range(agent_mcts.n_batches)])
    plt.plot(x, y_est, label=f"e=0", color='blue')
    plt.plot(x, y_true, color='blue', alpha=0.5)

    plt.xlabel('Simulations')
    plt.ylabel(f'Value s_{root_state_index}')
    plt.legend()
    plt.show()

    ##########################
    # Play games
    ##########################

    def play_game(agent, env, n_games=1, starting_state=5):
        """
        Run MCTS till either it reaches the goal state of fails.
        """
        returns = list()
        n_successes = 0
        for i_game in range(n_games):
            # Starting state
            s = starting_state
            s_check = agent.env.reset(start_state=s)
            assert s == s_check
            s_check = env.reset(start_state=s)
            assert s == s_check
            G = 0
            done = False

            print(f" Strating in {s}")

            while not done:
                actions, v_roots = agent.evaluate(n_mcts_runs=1, eval_range=[s], reset=False)
                # one experiment, so take that single action
                action = np.argmax(actions[0, s])
                (s_next, r, done, _) = env.step(action)

                print(f"Took {action} ---> ended up in {s_next}")
                env.render()

                G += r
                # if we reached a goal state
                if s_next in agent.goal_states:
                    n_successes += 1
                s = s_next
                assert s == env.s
            returns.append(G)

            print(f"Game {i_game} is done .. ")
            print()
            print()

        return n_successes, returns

    # How many times will we play the FrozenLake game with MCTS
    agent_mcts = AgentMCTS(env, discount_factor=DISCOUNT_FACTOR, uct_tree_policy=True, num_rollouts=80000, horizon=500,
                           uct_exploration_weight=50)
    n_games = 0

    start = time.time()

    n_successes, returns = play_game(agent_mcts, env, n_games=n_games, starting_state=EVAL_RANGE[0])

    end = time.time()
    print(f"Out of {n_games}, we succeeded {n_successes}. Accumulated discounted returns were {returns}. \n")
    print(f"All the games took {(end - start) // 60} mins {(end - start) % 60} secs")

