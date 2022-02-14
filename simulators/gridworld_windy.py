import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from gym.envs.toy_text import discrete
from utils.distribution import categorical_sample

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


class GridworldEnv(discrete.DiscreteEnv):
    """
    Gridworld simulator.
    """
    def is_terminal(self, s, t=None):
        return s in self.goal_states

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
            for j in range(i+1, n):
                D[i, j] = self.distance(states[i], states[j])
                D[j, i] = self.distance(states[i], states[j])

        return D

    def _set_goal_state(self, goal_state_coordinates):
        """
        Set a goal state for the gridworld.
        """
        if goal_state_coordinates:
            state_int_y = goal_state_coordinates[0]
            state_int_x = goal_state_coordinates[1]
            state_int = self.make_state_index(state_int_y, state_int_x)

        return state_int, np.array([state_int_x, state_int_y])

    def _add_disturbance_wind(
        self, P_temp, states_list, reward_wind, wind_rate, wind_direction
    ):
        """
        Disturbance:  no matter action you do. you transtion to the direction of the wind.
        Wind blows, you transition to the direction of the wind but diagonally clockwise or anti-clockwise.

        Choose your wind states CAREFULLY to they do not transition to the other part of the grid.

        E.G. [5, 5] Grid
            You are in the state 1, wind direction is  0,  thus you transition to the state MAX_COLS - 1
            You are in the state 19, wind direction is  2,  thus you transition to the state -MAX_COLS + 1
            You are in the state 19, wind direction is  3,  thus you transition to the state + 1 + MAX_COLS
        Thus, Direction 1 is southt-east, Direction 0 is north-east.

        Args:
            P_temp (int): mutubale argument. Gets updated in the function and then returned by the function.
            wind_direction (int): specify from where wind blows for the windy states (UP:1, RIGHT:1 etc)

        Returns:
            (dict) transition dictionary
        """
        for windy_state in states_list:
            y, x = self.make_coordinates(windy_state)
            is_done = lambda w_s: (w_s == self.goal_state_1) or (w_s == self.goal_state_0)
            reward = 0.0 if is_done(windy_state) else self.reward
            if wind_direction == 0:
                state_shift = +self.MAX_X - 1
            elif wind_direction == 1:
                state_shift = -1 - self.MAX_X
            elif wind_direction == 2:
                state_shift = -self.MAX_X + 1
            else:
                state_shift = +1 + self.MAX_X
            ### Transitions "windy"
            ns_up = windy_state if y <= 0 else windy_state - self.MAX_X
            ns_right = windy_state if x >= (self.MAX_X - 1) else windy_state + 1
            ns_down = windy_state if y == (self.MAX_Y - 1) else windy_state + self.MAX_X
            ns_left = windy_state if x == 0 else windy_state - 1
            # Note: no matter the action you get the reward_wind
            P_temp[windy_state][UP] = [
                (1.0 - wind_rate, ns_up, reward, is_done(ns_up)),
                (wind_rate, windy_state + state_shift, reward_wind, False),
            ]
            P_temp[windy_state][RIGHT] = [
                (1.0 - wind_rate, ns_right, reward, is_done(ns_right)),
                (wind_rate, windy_state + state_shift, reward_wind, False),
            ]
            P_temp[windy_state][DOWN] = [
                (1.0 - wind_rate, ns_down, reward, is_done(ns_down)),
                (wind_rate, windy_state + state_shift, reward_wind, False),
            ]
            P_temp[windy_state][LEFT] = [
                (1.0 - wind_rate, ns_left, reward, is_done(ns_left)),
                (wind_rate, windy_state + state_shift, reward_wind, False),
            ]

        return P_temp

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
        """
        Given a position (row, col) and an action a, return the resulting position (row, col).
        """
        # Left
        if a == 3:
            col = max(col - 1, 0)
        # Down
        elif a == 2:
            row = min(row + 1, self.nrow - 1)
        # Right
        elif a == 1:
            col = min(col + 1, self.ncol - 1)
        # Up
        elif a == 0:
            row = max(row - 1, 0)

        return row, col

    def __init__(self, shape=[5, 7], wind_rate_model=(None, None), t_max=100, seed=0, reward_wind=-1):
        """
        Creates an instance of GridWorldEnv.

        Need  to run .perform_reset() first for each episode.

        :param shape: (list/tuple) of length 2, with height and width of the grid
        :param t_max: (int) maximum time to act
        """
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError("shape argument must be a list/tuple of length 2")
        self.shape = shape
        # Number of rows
        self.MAX_Y = self.nrow = shape[0]
        # Number of columns
        self.MAX_X = self.ncol = shape[1]
        nS = np.prod(shape)
        # Number of actions is set
        nA = 4
        # Empty transition dictionary
        P = {}
        self.P = P
        # Reward Matrix S x A x S
        self.R = None
        # Time of an episode
        self.t = None
        # Maximum time when that an episode can last
        self.t_max = t_max
        # Number of time steps in NSMDP paper, we consider it as a possible disturbances within episode
        self.nT = 1
        # State transition matrix of shape S x A x nT x S
        self.T = None
        # Reward for regular tranistion
        self.reward = -1
        # Random or deterministic goal
        self.init_state = None
        self.init_state_coordinates = None
        self.goal_state_0, self.goal_state_0_coordinates = self._set_goal_state([0, 0])
        self.goal_state_1, self.goal_state_1_coordinates = self._set_goal_state([self.shape[0] - 1, self.shape[1] - 1])
        self.goal_states = [self.goal_state_0, self.goal_state_1]
        # isd is initial state distribution
        isd = np.ones(nS) / nS
        # Placeholders for wind == quicksands
        self.wind_rate_west = wind_rate_model[0]
        self.wind_west_states = []
        self.wind_rate_south = wind_rate_model[1]
        self.wind_south_states = []
        self.reward_wind = reward_wind
        # When is the value not a zero
        self.decision_epsilon = 0.0000001

        # Interesting, for experiments, starting states
        self.starting_states = [11, 7]  #  [11, 7, 3]

        # Set seeds
        super(GridworldEnv, self).__init__(nS, nA, P, isd)
        np.random.seed(seed)
        self.seed = seed

    def build(
        self, start_state=None, wind_rate=(None, None), reward_wind=None, seed=0
    ):
        """
        Like reset, but reinitilizes the whole environment so you can set new wind rates and create new R and T.
        Fill in grid world with starting state, goal state and transition dictionary/matrix.
        Wind is a stochastic event that wont allow you to move in the direction specified by your action.
        Args:
            wind_rate (0 <= float <= 1)  define how probable it is that wind blows you up
        Returns:
             coordinates x and y, and the goal state coordinates
        """
        self.t = 0
        self.s = start_state or self.reset()
        self.init_state = self.s
        self.init_state_coordinates = self.make_coordinates(self.s)
        self.reward_wind = reward_wind or self.reward_wind

        # Fill in transition dictionary
        P = {}
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.nA)}
            is_done = lambda s: s == self.goal_state_0 or s == self.goal_state_1
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
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
            it.iternext()

        # If the env was created with some other wind rates keep them - if no new given
        self.wind_rate_west = wind_rate[0] or self.wind_rate_west
        if self.wind_rate_west:
            # Select cells 'west' wind: windy_states_'s
            w_west_0 = 1  # 10  #  self.goal_state_0 + 1
            # w_west_1 = self.goal_state_0 + 2
            w_west_2 = 5  # 6  #  self.goal_state_0 + 1 + self.MAX_X
            # w_west_3 = self.goal_state_0 + 2 + self.MAX_X
            # w_west_4 = 2  # self.goal_state_0 + 1 + self.MAX_X * 2
            # w_west_5 = self.goal_state_0 + 2 + self.MAX_X * 2
            self.wind_west_states = [
                w_west_0,
                # w_west_1,
                w_west_2,
                # w_west_3,
                # w_west_4,
                # w_west_5,
            ]
            # Add 'west' wind
            P = self._add_disturbance_wind(
                P,
                self.wind_west_states,
                wind_rate=self.wind_rate_west,
                reward_wind=reward_wind or self.reward_wind,
                wind_direction=3,
            )

        # If the env was created with some other wind rates keep them - if no new given
        self.wind_rate_south = wind_rate[1] or self.wind_rate_south
        if self.wind_rate_south:
            # Select cells 'southern' wind: windy_states_'s
            w_south_0 = self.goal_state_1 - self.MAX_X * 1
            w_south_1 = self.goal_state_1 - self.MAX_X * 1 - 1
            # w_south_2 = self.goal_state_1 - self.MAX_X * 1 - 2
            # w_south_3 = self.goal_state_1 - self.MAX_X * 2
            # w_south_4 = self.goal_state_1 - self.MAX_X * 2 - 1
            # w_south_5 = self.goal_state_1 - self.MAX_X * 2 - 2
            self.wind_south_states = [
                w_south_0,
                w_south_1,
                # w_south_2,
                # w_south_3,
                # w_south_4,
                # w_south_5,
            ]
            # Add 'southern' wind
            P = self._add_disturbance_wind(
                P,
                self.wind_south_states,
                wind_rate=self.wind_rate_south,
                reward_wind=reward_wind or self.reward_wind,
                wind_direction=1,
            )

        self.P = P
        # Make sure to fill in the resulting matrices
        self.generate_transition_matrix()
        self.generate_reward_matrix()

        return self.make_coordinates(self.s), self.goal_state_0_coordinates

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
            actual_next_state = self.make_state_index(*self.inc(*self.make_coordinates(state), action))
            if next_state != actual_next_state:
                return model.reward_wind
            else:
                return reward

        # Fill in transition dictionary
        P = {}
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])
        while not it.finished:
            s = it.iterindex
            # y, x = it.multi_index
            P[s] = {a: [] for a in range(self.nA)}
            is_done = lambda s: s == self.goal_state_0 or s == self.goal_state_1
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
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
                # Possible next state
                row_p, col_p = self.inc(row, col, a)
                s_p = self.make_state_index(row_p, col_p)
                # We are at time 0 and with action a we can transition to s_p, only to s_p for now
                T[s, a, 0, s_p] = 1.0
                rs = self.reachable_from_s_a(s, a)
                nrs = sum(rs)
                if nrs == 1:
                    # We can only transition to a single state s_p for this action
                    T[s, a, :, :] = T[s, a, 0, :]
                else:
                    # More states reachable, redistribute that 1 prob. mass to all the possible ones
                    w0 = np.array(T[s, a, 0, :])
                    for transition in self.P[s][a]:
                        # Assign next state the prob of that state
                        w0[transition[1]] = transition[0]
                    T[s, a, :, :] = w0
                    # TODO: Here, NSMDP also creates a constraint on the changed transitions, so the do not change too
                    #       much within episode. Do I need this?

        self.T = T
        return T

    def generate_reward_matrix(self):
        """
        Create the reward matrix of shape S x A x nT x S for the environment model.
        """
        R = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float) + self.reward_wind

        for s in range(self.nS):
            for a in range(self.nA):
                for transition_tuple in self.P[s][a]:
                    R[s, a, 0, transition_tuple[1]] = transition_tuple[2]

        # In goal states we always have reward 0
        for goal_state in self.goal_states:
            R[goal_state] = 0
        self.R = R
        return R

    def perform_step(self, a):
        """
        Act with action `a` in the environment.

        :param a: (int) action to perform
        :return: coordinates-(x,y), reward, done flag, _
        """

        obs_tp1, reward, done, _ = self.step(a)
        self.t += 1
        if self.t == self.t_max:
            done = True

        return self.make_coordinates(obs_tp1), reward, done, _

    @space_decorator
    def render(self, mode="human", close=False):
        self._render(mode, close)

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = sys.stdout
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            if self.s == s:
                output = " x "
            elif s in self.goal_states:
                output = " T "
            elif s in self.wind_south_states or s in self.wind_west_states:
                output = " . "
            else:
                output = " o "
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()
            outfile.write(output)
            if x == self.shape[1] - 1:
                outfile.write("\n")
            it.iternext()


if __name__ == "__main__":

    env = GridworldEnv(shape=[5, 5])
    env.build(wind_rate=(0.1, 0.9))

    P = env.generate_transition_matrix()

    ####################################
    #       Generate experience        #
    ####################################

    experience = []

    # env.transition(0,0,0)

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

    print(experience)
    print(len(experience))
