from abc import ABC
from collections import defaultdict

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from gym.envs.toy_text import discrete

from simulators.gridworld_windy import GridworldEnv


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)


class ModelGridworld(GridworldEnv):

    def __init__(self, experience_tuples, T=None, shape=(5, 7), wind_states=[], reward_wind=-2, seed=0):
        """
        Create an evnironment from experience tuples or T (transition matrix)
        Args:
            experience_tuples (list): list of tuples of the format (state, a, reward, state_prime)
        """
        super().__init__(shape=shape, t_max=100, seed=seed, reward_wind=reward_wind)
        # Getting access to some function defined in the main environment
        self.experience_arr = np.array(experience_tuples)
        self.experience_dict = None
        # When is the value not a zero
        self.decision_epsilon = 0.0000001
        self.machine_epsilon = np.finfo(float).eps
        # If we are creating an environment from transition matrix, we do not need exp.
        self.T = T

    def _create_experience_dict(self):
        """
        'Utility' function that procesess list with experience tupples and creates a ditionary that stores
        counts of s-a pairs and counts of s.
        """
        experience_arr = self.experience_arr.copy()
        self.experience_dict = DeepDict(DeepDict(DeepDict(list)))

        # For each possible state and action
        for s in range(self.nS):
            for a in range(self.nA):
                s_a_experience = experience_arr[
                    np.where((experience_arr[:, 0] == s) * (experience_arr[:, 1] == a))
                ]
                self.experience_dict[s][a]["count"] = np.zeros(self.nS)
                # There is experience for this s, a
                if s_a_experience.shape[0] > 0:
                    unique, counts = np.unique(
                        s_a_experience[:, 3].astype(int), return_counts=True
                    )
                    self.experience_dict[s][a]["count"][unique] = counts

    def estimate_transition_matrix(self):
        """
        Generate transition matrix of shape S x A x S.
        Should be called before create_transition_dict.
        """
        # Experience dictionary was not yet created
        if self.experience_dict is None:
            self._create_experience_dict()

        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        for s in range(self.nS):
            for a in range(self.nA):
                # Binomial/Multinomial likelihood and some prior
                row, col = self.make_coordinates(s)
                row_p, col_p = self.inc(row, col, a)
                s_p = self.make_state_index(row_p, col_p)
                # Assume we know which state we 'normally' transition to given the action (informative prior)
                self.experience_dict[s][a]["count"][s_p] += 1
                # Assume uninformative prior
                #self.experience_dict[s][a]["count"][:] += 1
                # Assume spacial prior
                # rs = self.reachable_from_s(s)
                # self.experience_dict[s][a]["count"][rs] += 1
                binomial_likelihood = self.experience_dict[s][a]["count"] / (self.experience_dict[s][a]["count"].sum() + self.machine_epsilon)
                T[s, a, 0, :] = binomial_likelihood

        self.T = T
        return T

    def create_transition_dict(self):
        """
        Should be called after estimate_transition_matrix.
        """
        # Check if we have a transition matrix to work with
        if self.T is None or sum(sum(sum(sum(self.T)))).sum() < 1:
            raise Exception("Transistion matrix is not yet created OR all entries are zeros.")

        def _check_reward(model, state, next_state, action):
            """
            Function that determines the rewards: we assume that the reward function is known and that
            if there is no natural/regular/usual transistion (some reward) than "adverserial"/bad transition happen
            """
            if next_state not in self.reachable_from_s(state):
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


if __name__ == "__main__":

    GRID_SHAPE = [5, 5]

    # Create true model
    env = GridworldEnv(shape=GRID_SHAPE, seed=None)
    env.perform_reset(wind_rate=(0.9, 0.00001))
    T_true = env.generate_transition_matrix()

    # Learn model for different number of experience tuples
    experience_volumes = [1000, 2000]
    # Make plot to see if there is convergence
    errs = []
    big_errs_count = []
    # Increase experience, see if our transition function gets better.
    for n_train_episodes in experience_volumes:

        ####################################
        #        Generate experience       #
        ####################################

        experience = []
        for episode in range(n_train_episodes):
            # TODO: Now starting only in windy states: gives way better estimation of transition matrix
            s_w = np.random.choice(env.wind_south_states + env.wind_west_states, 1)[0]
            # s_w = np.random.choice(range(env.nS), 1)[0]
            state, _ = env.perform_reset(start_state=s_w, seed=None)
            state = env.make_state_index(state[0], state[1])
            t = 0
            done = False
            while not done:
                # Usually, we need an algorithm that takes in the current state
                a = np.random.choice([0, 1, 2, 3])
                (x, y), reward, done, _ = env.perform_step(a)
                state_prime = env.make_state_index(x, y)
                # Record experience
                experience.append((state, a, reward, state_prime))
                state = state_prime
                t += 1

        ####################################
        #         Learn the model          #
        ####################################

        model = ModelGridworld(
            experience_tuples=experience,
            wind_states=env.wind_south_states + env.wind_west_states,
            reward_wind=-2,
            shape=GRID_SHAPE,
            seed=None,
        )
        T = model.estimate_transition_matrix()
        P = model.create_transition_dict()

        #######################################
        # Check how well we learnt the model  #
        #######################################

        # Check if all the non-zero entries in T are also non-zero in T_true i.e. "total probability mass error"
        divergence_value = sum(sum(abs(T_true - T))).sum() - 8
        T_masked = ma.masked_where(T > 0, T)
        T_true_masked = ma.masked_where(T_true > 0, T_true)
        print("All the non-zero entries in T_estimated are also non-zero in T_true: ",
              np.array_equal(T_masked.mask, T_true_masked.mask))

        # See which transition probabilities are different by more than 1 p.p.
        count = 0
        for state in range(env.nS):
            for action in range(env.nA):
                if (abs(T_masked[state, action, 0, :].data - T_true_masked[state, action, 0, :].data).sum() < 0.01):
                    pass
                else:
                    count += 1
                    # If there was a lot of experience but we still were making a lot errors
                    if n_train_episodes > 30000 and count > 12:
                        print(state, " | ", action)
                        print(T[state, action, 0, :])
                        print()

        errs.append(divergence_value)
        big_errs_count.append(count)

    print(f"For each experience volume {experience_volumes}. Summed differences between T_estimated and T_true: \n", errs)
    print(f"How many entries in T_estimated and T_true differ by more than 1 p.p.: \n", big_errs_count)

    ########################
    #        Results       #
    ########################

    ## For 6 "disturbance states":
    #   In the past, I reached total probability mass error of 0.2836 for [5,5] grid on 150000 train episodes of experience.
    #              the count was 11.

    #   In the past, I reached total probability mass error of 0.194 for [5,5] grid on 200000 train episodes of experience.
    #              the count was 4.
