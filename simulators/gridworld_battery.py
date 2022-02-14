import sys
from collections import defaultdict

from gym.envs.toy_text import discrete

from utils.plotting import plot_gridworld_value
from agents.agent_dp import AgentDP
from simulators.gridworld_windy import GridworldEnv
from utils.model_space import *


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)


def space_decorator(func):
    """
    Put an empty line after the func is executed.
    """
    def inner1(*args, **kwargs):
        func(*args, **kwargs)
        print("")

    return inner1


class BatteryGridworldEnv(GridworldEnv):
    """
    Gridworld simulator but adjusted to imitate stochastic transition in price.
    """

    def __init__(self, shape=[5, 7], wind_rate_model=(None, None), t_max=100, seed=0, reward_wind=-1):
        """
        Creates an instance of GridWorldEnv.
        Need  to run .perform_reset() first.
        Args:
            shape (list/tuple): of length 2, with height and width of the grid
            t_max (int): maximum time to act in a single episode
        """
        super(BatteryGridworldEnv, self).__init__(shape, wind_rate_model, t_max, seed, reward_wind)
        # Interesting, for testing riskiness, starting states
        self.starting_states = [0, 19, 20, 28]

        np.random.seed(seed)

        # Random or deterministic goal
        self.init_state = 19
        self.init_state_coordinates = (2, 5)
        self.goal_state_0, self.goal_state_0_coordinates = self._set_goal_state([0, self.shape[1] - 1])
        self.goal_state_1, self.goal_state_1_coordinates = self._set_goal_state([self.shape[0] - 1, self.shape[1] - 1])
        self.goal_state_2, self.goal_state_2_coordinates = self._set_goal_state([2, 0])
        self.goal_states = [self.goal_state_0, self.goal_state_1, self.goal_state_2]

    def build(self, start_state=None, wind_rate=(0.0, 0.0), reward_wind=None, seed=0):
        """
        Fill in grid world with starting state, goal state and transition dictionary/matrix.
        In this grid world, transitions UP/DOWN are stochastic, while LEFT/RIGHT deterministic.
        Args:
            wind_rate (tuple with elements 0 <= float <= 1): first element for prob of transitioning to the state
            above when action UP is performed, second element of the tuple the same but for DOWN action
        Returns:
             tuple, of (coordinates x and y, the goal state coordinates)
        """
        self.t = 0
        self.s = self.init_state if start_state is None else start_state
        self.init_state = self.s
        self.init_state_coordinates = self.make_coordinates(self.s)
        self.reward_wind = reward_wind or self.reward_wind
        self.wind_rate_west = wind_rate[0] or self.wind_rate_west
        self.wind_rate_south = wind_rate[1] or self.wind_rate_south

        # Fill in transition dictionary
        P = {}
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.nA)}
            is_done = lambda s: s in [self.goal_state_0, self.goal_state_1 , self.goal_state_2]
            reward = 0.0 if is_done(s) else self.reward
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - self.MAX_X
                ns_right = s if x == (self.MAX_X - 1) else s + 1
                ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0-self.wind_rate_west, ns_up, reward, is_done(ns_up)), (self.wind_rate_west, s, self.reward_wind, is_done(ns_up))]
                P[s][DOWN] = [(1.0-self.wind_rate_south, ns_down, reward, is_done(ns_down)), (self.wind_rate_south, s, self.reward_wind, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
            it.iternext()

        self.P = P
        # Make sure to fill in the resulting matrices
        self.generate_transition_matrix()
        self.generate_reward_matrix()

        return self.make_coordinates(self.s), self.goal_state_0_coordinates


if __name__ == "__main__":
    env = BatteryGridworldEnv(shape=[5, 7])
    env.build(wind_rate=(0.7, 0.7))
    T = env.generate_transition_matrix()

    agent = AgentDP(env)
    policy_true, v_true = agent.train_model_based(env=env)
    plot_gridworld_value(v_true.reshape(env.shape), env, true_model=True)

    print()

    ####################################
    #       Generate experience        #
    ####################################

    experience = []
    # Set the number of episodes
    n_train_episodes = 100
    # Generate an episode
    for episode in range(n_train_episodes):
        state, _ = env.build(seed=int(np.random.random() * 100))
        state = env.make_state_index(state[0], state[1])
        t = 0
        done = False
        while not done:
            # Usually, we need an algorithm that takes in  the current state
            a = np.random.choice([0, 1, 2, 3])
            (x, y), reward, done, _ = env.perform_step(a)
            state_prime = env.make_state_index(x, y)

            # Record experience
            experience.append((state, a, reward, state_prime))
            state = state_prime
            t += 1

    # print(experience)
    # print(len(experience))
