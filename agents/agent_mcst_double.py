import random
from math import sqrt, log
from collections import defaultdict
import time

from gym import spaces
import numpy as np

from utils.utils import assert_types
from simulators.gridworld_windy import GridworldEnv
from simulators.gridworld_battery import BatteryGridworldEnv
from utils.plotting import plot_gridworld_value
from agents.agent_dp import AgentDP
from utils.model_space import wasserstein_worstcase_distribution_analytical


SEED = None


def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    else:
        raise NotImplementedError


class AgentMCTS(object):
    """
    MCTS agent with random or UCT rollout policy.
    """

    def train(self, *args, **kwargs):
        """
        Raw fix: added this function to make it work with run batttery.
        """
        return 0

    def get_epsilon(self, *args, **kwargs):
        """
        Raw fix: added this function to make it work with run batttery.
        """
        return 0

    @staticmethod
    def uct_tree_policy(children, Q, N, ucb_constant=7):
        def _ucb(node):
            """
            Upper Confidence Bound of a chance node
            """
            return Q[node] + ucb_constant * sqrt(log(N[(node[0],)]) / N[node])

        q_vals = {child: _ucb(child) for child in children}

        return max(q_vals, key=q_vals.get), q_vals

    @staticmethod
    def random_tree_policy(children, Q, *args, **kwargs):
        return random.choice(children), {child: Q[child] for child in children}

    @staticmethod
    def softmax_policy(s, Q, tau=0.1, nA=4):
        exp_q_s = np.array([np.exp(Q[(s, a)]/tau) for a in range(nA)])
        exp_q_sum = sum(exp_q_s)
        return np.random.choice(range(nA), p=exp_q_s/exp_q_sum)

    def __init__(self, env, num_rollouts=500, horizon=100,
                 discount_factor=0.99, uct_exploration_weight=1,
                 uct_tree_policy=True, **kwargs):
        # If action space discrete we can work further if not stop
        if type(env.action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(env.action_space))
        else:
            raise TypeError("Action space is not descrete")
        self.env = env
        self.n_actions = len(self.action_space)
        self.num_rollouts = num_rollouts
        self.horizon = horizon
        self.discount_factor = discount_factor
        # Total reward of each node
        self.Q = defaultdict(float)
        # Total visit count for each node
        self.N = defaultdict(int)
        # Children of each node
        self.children = dict()
        self.uct_exploration_weight = uct_exploration_weight
        self.tree_policy = self.uct_tree_policy if uct_tree_policy else self.random_tree_policy

    def reset(self, p=None):
        """
        Reset the attributes if you want to plan for a new state.
        Expect to receive them in the same order as init.

        p : list of parameters
        """
        if p is None:
            self.__init__(self.action_space)
        else:
            assert_types(p, [spaces.discrete.Discrete, int, int, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4])

    def mcts_procedure(self, done):
        """
        Compute the entire MCTS procedure wrt to the selected tree policy.
        """
        Q_0 = defaultdict(float)
        Q_1 = defaultdict(float)
        N_0 = defaultdict(int)
        N_1 = defaultdict(int)
        # Children of each node
        children_0 = defaultdict(list)
        children_1 = defaultdict(list)
        unexplored_actions_0 = defaultdict(lambda: random.sample(list(range(self.env.nA)), self.env.nA))
        unexplored_actions_1 = defaultdict(lambda: random.sample(list(range(self.env.nA)), self.env.nA))

        # Testing/debugging
        self.v = np.zeros(self.env.nS)

        root_state = self.env.s
        for i_rollout in range(self.num_rollouts):
            # Rewards collected along the tree for the current rollout (path)
            rewards = list()
            terminal = done
            # Last chance node
            ch_node = (None, None)
            # Last decision node
            dc_node = (root_state,)
            # Keep track of visited nodes
            path = [ch_node, dc_node]
            # Hashing the path in order to see how much we explored on that branch of the tree
            path_tuple = tuple(path)
            root_path = path
            # Current node
            node = dc_node

            # Double MCTS :P
            flag = np.random.randint(2)
            if flag:
                unexplored_actions = unexplored_actions_0
                children = children_0
                Q = Q_0
                N = N_0
                Q_update = Q_1
                N_update = N_1
            else:
                unexplored_actions = unexplored_actions_1
                children = children_1
                Q = Q_1
                N = N_1
                Q_update = Q_0
                N_update = N_0

            # Selection
            select = True
            counter = 0
            while select:
                assert len(node) == 1
                if self.env.is_terminal(node[0]):
                    select = False
                else:
                    # Using the hashed path to see if have explored everything in this tree
                    if len(unexplored_actions[path_tuple]) > 0:
                        select = False
                    else:
                        ch_node, Q_sa = self.tree_policy(children[node], Q, N, self.uct_exploration_weight)
                        state_p, reward, terminal = self.env.transition(ch_node[0], ch_node[1])
                        path.append(ch_node)
                        rewards.append(reward)
                        dc_node = (state_p,)
                        path.append(dc_node)
                        # Hashing the path in order to see how much we explored on that branch of the tree later
                        path_tuple = tuple(path)
                        node = dc_node

            # Expansion
            assert len(node) == 1
            if not terminal:
                ch_node = (node[0], unexplored_actions[path_tuple].pop())
                children[node].append(ch_node)
                path.append(ch_node)
                state_p, reward, terminal = self.env.transition(ch_node[0], ch_node[1])
                rewards.append(reward)
                dc_node = (state_p,)
                path.append(dc_node)
                node = dc_node

            # Evaluation
            assert len(node) == 1
            t = 0
            assert "reward" in locals(), f'{path}    |   {terminal}   |    {node}'
            estimate = rewards.pop()
            state = node[0]
            while (not terminal) and (t < self.horizon):
                # Running MC rollout with the random policy
                action = self.env.action_space.sample()
                state, reward, terminal = self.env.transition(state, action)
                estimate += reward * (self.discount_factor ** t)
                t += 1

            # Debugging
            # if i_rollout % 1000 == 0:
            #
            #     print(path)
            #     print(estimate)
            #     print([Q[(root_state, a)] for a in range(self.env.nA)])

            # Backpropagation
            assert len(path) // 2 - 2 == len(rewards)
            while True:
                # Next decision node wrt chance node
                dc_node = path.pop()
                assert dc_node == node
                ch_node = path[-1]
                # Usually you do updates to the counts in backprop
                N_update[ch_node] += 1
                N_update[(ch_node[0], )] += 1
                assert (len(ch_node) == 2)
                Q_update[ch_node] += (estimate - Q_update[ch_node]) / N_update[ch_node]  # N[(ch_node[0],)]
                if len(rewards) != 0:
                    estimate = rewards.pop() + self.discount_factor * estimate
                    previous_ch_node = path.pop()
                    node = path[-1]
                else:
                    assert ch_node[0] == root_state
                    break

            # Assign back
            if flag:
                unexplored_actions_0 = unexplored_actions
                children_0 = children
                Q_0 = Q
                N_0 = N
                Q_1 = Q_update
                N_1 = N_update
            else:
                unexplored_actions_1 = unexplored_actions
                children_1 = children
                Q_1 = Q
                N_1 = N
                Q_0 = Q_update
                N_0 = N_update

        # Get action and value of the root state
        returns_root_state_0 = np.array([Q_0[(root_state, a)] for a in range(self.env.nA)])
        returns_root_state_1 = np.array([Q_1[(root_state, a)] for a in range(self.env.nA)])
        returns_root_state = (returns_root_state_1 + returns_root_state_0 )/ 2

        # Testing/debugging
        # for ch_node in list(Q.keys()):
        #     if ch_node != (None, None):
        #         self.v[ch_node[0]] = np.max([Q[(ch_node[0], i)] for i in range(self.env.nA)])
        # print(f"State: {root_state} | {returns_root_state}")
        # np.set_printoptions(precision=2)
        # self.v.reshape((5, 5))

        return np.argmax(returns_root_state), max(returns_root_state)

    def select_action(self, done=False, *args, **kwargs):
        """
        Run MCTS for the current env.s
        """
        return self.mcts_procedure(done)

    def evaluate(self, n_mcts_runs=1, parallel=False, eval_range=None):
        """
        Get an estimate of the value function for all states in the environment when each action is selected with MCTS.
        """
        policy = np.zeros((1, self.env.nS, self.env.nA))
        values = np.zeros((1, self.env.nS))

        def _evauate_policy_value(agent, eval_range):
            actions_experiments = np.zeros((agent.env.nS, agent.env.nA))
            decision_node_values = np.zeros(agent.env.nS)
            if eval_range is None:
                eval_range = range(1, agent.env.nS-1)  # agent.env.starting_states  #
            for s_init in eval_range:
                _ = agent.env.reset(start_state=s_init)
                action, root_value = agent.select_action(agent.env.is_terminal(agent.env.s))
                actions_experiments[s_init, action] = 1
                decision_node_values[s_init] = root_value

            return actions_experiments, decision_node_values

        if not parallel:
            for i in range(n_mcts_runs):
                actions_experiments, decision_node_values = _evauate_policy_value(self, eval_range)
                policy = np.vstack([policy, actions_experiments[np.newaxis, :, :]])
                values = np.vstack([values, decision_node_values[np.newaxis, :]])
        else:
            raise NotImplementedError("TODO: paralel implementation?")

        return policy[1:, :, :], values[1:, :]


def evalute_mcts(agent, env, wind_rate_model=None, seed=None, save_path=None):  # "./storage/"):
    # Get the true value function
    agent_true = AgentDP(env)
    policy, v_true = agent_true.train_model_based(env=env)
    plot_gridworld_value(v_true.reshape(env.shape), env, true_model=True, policy=policy, save_path=save_path)
    # Solving VI with matrix algebra
    #v_true = agent_true.train_model_based_matrix(env=env)
    #plot_gridworld_value(v_true.reshape(env.shape), env, true_model=True, save_path=save_path)

    # Create value function with mcts procedure
    n_mcts_runs = 1
    actions, v_roots = agent.evaluate(n_mcts_runs, eval_range=[2, 6, 10])  #
    average_policy = actions.sum(axis=0) / n_mcts_runs
    v_roots = v_roots.mean(axis=0)
    plot_gridworld_value(v_roots.reshape(env.shape), env, policy=average_policy, save_path=save_path)


if __name__ == '__main__':
    start = time.time()

    seed = None
    # Maximum length of an episode
    t_max = 100000

    # Which envs to run
    run_windy_gridworld = True
    run_battery_gridworld = False

    if run_windy_gridworld:
        # Create an environment and train the agent
        wind_rate_model = (0.8, 0.0)
        env = GridworldEnv([5, 5], wind_rate_model=wind_rate_model, t_max=t_max, seed=seed, reward_wind=-2)
        _, _ = env.build(start_state=13, seed=None)

        # Define what you need to do modelling
        agent_mcts = AgentMCTS(env, uct_tree_policy=True, num_rollouts=20000, horizon=15, discount_factor=.99,
                               uct_exploration_weight=1)
        evalute_mcts(agent_mcts, env, seed=None)

    if run_battery_gridworld:
        # Create an environment and train the agent
        up_down_stochasticity = (0.75, 0.75)
        env = BatteryGridworldEnv(shape=[5, 7], wind_rate_model=up_down_stochasticity, seed=seed, reward_wind=-1)
        env.build(wind_rate=up_down_stochasticity, seed=None)
        T = env.generate_transition_matrix()

        # Define what you need to do modelling
        agent_mcts = AgentMCTS(env, uct_tree_policy=True, num_rollouts=10000, horizon=2, discount_factor=.99,
                               uct_exploration_weight=1)
        evalute_mcts(agent_mcts, env, seed=None)

    end = time.time()
    print(f"{(end - start) // 60} mins {(end - start) % 60} secs")
