from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
# Do not delete this line with Axes3D eventhough it is not 'used'
#   actually used later for the 3D plot
from mpl_toolkits.mplot3d import Axes3D

from simulators.gridworld_windy import GridworldEnv


class AgentQLearning:
    """
    Reimplemented from here:
        https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb

    Check out double Q learning here:
        https://www.datahubbs.com/double-q-learning/#Double-Q-Function
    """

    def __init__(
        self,
        env,
        buffer=None,
        lr=None,
        gamma=None,
        epsilon_delta=None,
        epsilon_min=None,
        epsilon=1,
        loss_function=None,
        optimizer=None,
        alpha=0.1,
    ):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while following an epsilon-greedy policy
        Needs environment to determine how many actions there is.

        Args:
            env: OpenAI environment.
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        self.env = env
        self.replay_buffer = buffer
        self.learning_rate = lr
        self.alpha = alpha
        self.gamma = self.discount_factor = gamma
        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.epsilon_min = epsilon_min
        # The final action-value function: a nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.nA = self.env.action_space.n
        # Array used to see how well we trained
        self.stats = None
        self.id = "Q-Learning"

    @staticmethod
    def get_epsilon(it):
        """
        TODO: how to select this epsilon decay parameters?
        """
        return 1 - (it * 0.00095) if it < 1000 else 0.05

    @staticmethod
    def make_epsilon_greedy_policy(Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += 1.0 - epsilon
            return A

        return policy_fn

    def select_action(self, state, epsilon=0.0):
        policy = self.make_epsilon_greedy_policy(self.Q, epsilon=epsilon, nA=self.nA)
        action_probs = policy(state)

        return np.random.choice(np.arange(len(action_probs)), p=action_probs)

    def train(self, batch_size):
        """
        Determine policy and Q-funtion by doing backward updates on exprience tuples.

        Our behaviour policy is determined by experience buffer.
        Exploration (optimization) policy is fully greedy (Q-Learning).
        """
        # Don't learn without some decent experience
        if len(self.replay_buffer) < batch_size:
            return None

        # Random transition batch is taken from experience replay memory
        transitions = self.replay_buffer.sample(batch_size)
        # Transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, reward, next_state, done = transitions

        # Check if we have correct buffer output
        assert len(state) == len(action) == len(reward) == len(next_state) == len(done)

        # Update policy and Q-value funtion
        for state_i, action_i, reward_i, next_state_i, done_i in zip(
            state, action, reward, next_state, done
        ):
            # TD Update
            best_next_action = np.argmax(self.Q[next_state_i])
            td_target = (reward_i + self.discount_factor * self.Q[next_state_i][best_next_action])
            td_delta = td_target - self.Q[state_i][action_i]
            self.Q[state_i][action_i] += self.alpha * td_delta

        # As a standard `train()` should return the loss, now I return sum of q-values
        return sum(self.Q.values())

    def train_model_based(self, env, num_episodes=100000, discount_factor=1.0, alpha=0.01, epsilon=0.1,
                          seed=None, double_qlearning=False):
        """
        Do training on the environemnt

        From here: https://www.datahubbs.com/double-q-learning/#Double-Q-Function

        Args:
            env: OpenAI environment.
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        # Set the seed for reproducability
        env.seed = seed

        if not double_qlearning:
            # The final action-value function.
            # A nested dictionary that maps state -> (action -> action-value).
            Q = np.zeros((env.observation_space.n, env.action_space.n))
            ep = 0

            for i_episode in range(num_episodes):
                s = env.reset()
                done = False

                while not done:
                    # Greedy policy
                    if np.random.rand() < epsilon:
                        action = env.action_space.sample()
                    else:
                        max_qs = np.where(np.max(Q[s]) == Q[s])[0]
                        action = np.random.choice(max_qs)

                    s_prime, r, done, _ = env.step(action)
                    Q[s, action] += alpha * (r + discount_factor * np.max(Q[s_prime]) - Q[s, action])

                    s = s_prime
                    if done:
                        break

            for s in range(env.nS):
                for a in range(env.nA):
                    self.Q[s][a] = Q[s, a]

        else:
            # Initialize Q-table
            Q1 = np.zeros((env.observation_space.n, env.action_space.n))
            Q2 = Q1.copy()
            i_episode = 0

            for i_episode in range(num_episodes):
                s = env.reset()
                done = False

                while not done:
                    # Greedy policy
                    if np.random.rand() < epsilon:
                        action = env.action_space.sample()
                    else:
                        q_sum = Q1[s] + Q2[s]
                        max_q = np.where(np.max(q_sum) == q_sum)[0]
                        action = np.random.choice(max_q)
                    s_prime, r, done, _ = env.step(action)

                    # Flip a coin to update Q1 or Q2
                    if np.random.rand() < 0.5:
                        Q1[s, action] += alpha * (r +
                                                    discount_factor * Q2[s_prime, np.argmax(Q1[s_prime])] - Q1[s, action])
                    else:
                        Q2[s, action] += alpha * (r +
                                                    discount_factor * Q1[s_prime, np.argmax(Q2[s_prime])] - Q2[s, action])

                    s = s_prime
                    if done:
                        break

            Q = (Q1 + Q2) / 2
            for s in range(env.nS):
                for a in range(env.nA):
                    self.Q[s][a] = Q[s, a]

        def _to_V(Q):
            """
            Make V values out of Q values.
            """
            V = [np.max(Q[i]) for i in range(0, env.nS)]
            # V = np.array(V).reshape(env.shape)

            return np.array(V)

        def _to_policy_arr(Q):
            """
            Make policy array (as previously in DP) - we have Q values we can take an argmax from them.

            Args:
                Q (dict): dictionary where keys are state indicies and values are numpy arrays of shape (nA,)
            """
            # Policy is your Q function
            dict_vals = dict(sorted(Q.items())).values()
            policy_arr = np.vstack(list(dict_vals))
            # Policy derived for your V function

            return policy_arr

        return _to_policy_arr(self.Q), _to_V(self.Q)

    def evaluate(self, eval_env, policy, high_neg_reward, starting_states=[3, 7, 8, 12, 13]):
        """
        :param policy: array of the shape S x A  with probabilties identifing which action to take where
        :return:
        """
        # Count the number of times, extreme negatives rewards appeared
        n_high_neg_rewards = 0
        returns = 0
        for s_t in starting_states or range(eval_env.nS):
            _, _ = eval_env.perform_reset(start_state=s_t)
            done = False
            while not done:
                # print(s_t)
                a = np.argmax(policy[s_t])
                s_t_1, reward, done, _ = eval_env.step(a)
                returns += reward
                if high_neg_reward == reward:
                    n_high_neg_rewards += 1
                s_t = s_t_1

        return returns, n_high_neg_rewards


if __name__ == "__main__":

    # Create a grid of wind values on which I am going to plot the
    g = np.mgrid[0:1:0.1, 0:1:0.1]  # np.mgrid[0:1.1:0.1, 0:1.1:0.1]
    wind_rates_mesh = list(zip(*(x.flat for x in g)))
    epsilon = 0.00001

    results_grid = np.zeros((1, 6))
    for idx, wind_rate_true in enumerate(
        [(0.9, 0.0001)]):  # wind_rates_mesh):
        # Not sure if my DP works with
        wind_rate_true = (
            np.array(wind_rate_true) + epsilon
            if ((wind_rate_true[0] < 1.0) & (wind_rate_true[1] < 1.0))
            else (1 - epsilon, 1 - epsilon)
        )
        V_sum = 0
        policy_sum = 0
        model_returns_sum = 0
        perfect_returns_sum = 0
        model_num_high_neg_rewards_sum = 0
        perfect_num_high_neg_rewards_sum = 0

        for i, seed in enumerate([0]):  # enumerate([0, 10, 13, 17, 23, 42]):
            # Set seed
            np.random.seed(seed)
            # Maximum length of an episode
            t_max = 100
            # Create an environment and train the agent
            env = GridworldEnv([5, 5], t_max=t_max, seed=seed)
            wind_rate_model = (0.0000001, 0.0000001)

            # Define what you need to do model
            agent_qlearning = AgentQLearning(env)

            _, _ = env.perform_reset(start_state=13, wind_rate=wind_rate_model, seed=seed)
            policy, V = agent_qlearning.train_model_based(env)

            ####### TESTING #######
            plot_gridworld_value((V).reshape(env.shape), env)
            plt.show()
            #######################

            # Evaluate the policy on the true env
            _, _ = env.perform_reset(start_state=13, wind_rate=wind_rate_true, seed=seed)
            model_returns, model_num_high_neg_rewards = agent_qlearning.evaluate(
                env, policy, high_neg_reward=env.reward_wind
            )

            # Agent knows the environment: get perfect policy and return
            policy_upper_bound, V_perfect_knowledge = agent_qlearning.train_model_based(env)
            perfect_returns, perfect_num_high_neg_rewards = agent_qlearning.evaluate(
                env, policy_upper_bound, high_neg_reward=env.reward_wind
            )

            ##### TESTING #######
            plot_gridworld_value((V_perfect_knowledge).reshape(env.shape), env)
            plt.show()
            #####################

            V_sum += V
            policy_sum += policy
            model_returns_sum += model_returns
            perfect_returns_sum += perfect_returns
            model_num_high_neg_rewards_sum += model_num_high_neg_rewards
            perfect_num_high_neg_rewards += perfect_num_high_neg_rewards

        ######################################
        # Store data for 3D visulaiztion     #
        ######################################

        # How many seeds we had
        n = i + 1
        result_row = np.hstack(
            [wind_rate_true, np.array([model_returns_sum / n, perfect_returns_sum / n])]
        )
        # For now do not divide by n, though might be better to do it
        result_row = np.hstack(
            [
                result_row,
                np.array(
                    [model_num_high_neg_rewards_sum, perfect_num_high_neg_rewards_sum]
                ),
            ]
        )
        results_grid = np.vstack([results_grid, result_row])

        # Log some results
        print("policy return under assumed env: ", result_row[1])
        print("policy return under true env : ", result_row[2])
        # print()

    # Plot rewards
    x = results_grid[1:, 0]
    y = results_grid[1:, 1]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(x, y, results_grid[1:, 2], linewidth=0.2, antialiased=True)
    ax.plot_trisurf(
        x, y, results_grid[1:, 3], linewidth=0.2, antialiased=True, alpha=0.5,
    )
    plt.title("Rewards, DP agent with imperfect knowledge (blue) vs perfect (red)")
    ax.set_xlabel("wind_west")
    ax.set_ylabel("wind_south")
    ax.set_zlabel("rewards")
    plt.show()

    x = results_grid[1:, 0]
    y = results_grid[1:, 1]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(x, y, results_grid[1:, 4], linewidth=0.2, antialiased=True)
    ax.plot_trisurf(
        x, y, results_grid[1:, 5], linewidth=0.2, antialiased=True, alpha=0.5,
    )
    plt.title("High negative rewards.")
    ax.set_xlabel("wind_west")
    ax.set_ylabel("wind_south")
    ax.set_zlabel("rewards")
    plt.show()

    print()