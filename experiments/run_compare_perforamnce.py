import os
import logging
from collections import defaultdict
import time
from copy import copy, deepcopy

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import plot_gridworld_value
from utils.model_space import wasserstein_worstcase_distribution_analytical
from agents.agent_dp import AgentDP
from agents.agent_mcts import AgentMCTS
from agents.agent_rmcts_batched import AgentRMCTSBatched
from agents.agent_rmcts_iter import AgentRMCTSIterative
from simulators.gridword_frozen_lake import FrozenLakeEnv
from simulators.gridworld_lava import LavaWorld
# For debugging
from utils.distribution import *
from utils import plotting
import pickle


def run(agent, env, eval_range, n_experiments, save_path=None):
    """
    Run one (evauation) experiment to derive the value function and policy for the root state (eval_state).

    Returns:
        policy_est (numpy.array): dimensions [num_experiments, nS, nA]
        v_est (numpy.array):  dimensions [num_experiments, nS]
        v_est_batches (numpy.array): dimensions [num_experiments, num_eval_batches, nS]
         v_true_batches (numpy.array): dimensions [num_experiments, num_eval_batches, nS]
    """
    # The actual state being evaluated will be set during mcts evaluation
    env.reset(start_state=eval_range[0])
    # Set-up the environment
    env.generate_transition_matrix()
    env.generate_reward_matrix()
    env.render(pretty_print=False)

    # Get the policy and value function
    start = time.time()
    # Create value function with mcts procedure
    policy_est, v_est, v_est_batches, v_true_batches = agent.evaluate(n_experiments, eval_range=eval_range)
    end = time.time()
    logging.info(f"Finished evaluating {len(eval_range)} state(s) in {(end - start) // 60} mins {(end - start) % 60} secs")

    if save_path:
        f = open(os.path.join(save_path, env.env_id + '_epsilon_' + str(epsilon_robust) + '_gamma_'
                              + str(discount_factor), agent.agent_id, 'agents_P.pkl'), "wb")
        pickle.dump(agent.T_robust, f)
        f.close()

    return policy_est, v_est, v_est_batches, v_true_batches


def play_game(agent_policy, env, discount_factor, starting_state, n_games=1):
    """
    Run MCTS till either it reaches the goal state of fails.
    """
    discounted_returns = list()
    n_successes = 0

    k = 0
    for i_game in range(n_games):
        # Starting state
        s = starting_state
        s_check = env.reset(start_state=s)
        assert s == s_check
        s_check = env.reset(start_state=s)
        assert s == s_check

        G = 0
        done = False
        t = 0
        while not done:
            action = np.random.choice([0, 1, 2, 3], p=agent_policy[s])
            (s_next, r, done, _) = env.step(action)

            G += r * (discount_factor ** t)
            t += 1
            # if we reached a goal state
            if s_next in env.goal_states:
                n_successes += 1
            s = s_next
            assert s == env.s

        discounted_returns.append(G)

    return n_successes, discounted_returns


if __name__ == "__main__":

    SEED = 20
    DISCOUNT_FACTOR = 0.95
    SAVE_PATH = "storage/results_report"  # None

    logging.basicConfig(filename='example.log', level=logging.DEBUG)

    # TODO: introduce a config file or/and pass as cmd args
    n_experiments = 1
    discount_factor = DISCOUNT_FACTOR
    eval_range = [0]  # + [1, 5] + list(range(7, 30)) + [33, 34]
    save_path = SAVE_PATH
    load_path = SAVE_PATH
    num_rollouts = 120000
    # Evaluate after how many simulations
    n_evaluation = 40000
    num_eval_batches = num_rollouts // n_evaluation
    epsilon_robust = 0.2
    uct_exploration_weight = 50
    # How many times to 'play' with one model setting
    n_simulation = 10000

    # What environment to use
    env = LavaWorld()  # FrozenLakeEnv()  #
    env.generate_transition_matrix()
    env.generate_reward_matrix()

    # Define what you need to do modelling
    agent_mcts = AgentMCTS(env, num_rollouts=num_rollouts, n_evaluation=n_evaluation,
                           discount_factor=discount_factor, uct_tree_policy=True, horizon=500,
                           uct_exploration_weight=uct_exploration_weight)
    agent_rmcts_batched = AgentRMCTSBatched(env, num_rollouts=num_rollouts, num_batches=num_eval_batches,
                                            discount_factor=discount_factor, uct_tree_policy=True, horizon=500,
                                            uct_exploration_weight=uct_exploration_weight, epsilon_robust=epsilon_robust)
    agent_rmcts_iter = AgentRMCTSIterative(env, num_rollouts=num_rollouts, n_evaluation=n_evaluation,
                                           discount_factor=discount_factor, uct_tree_policy=True, horizon=500,
                                           uct_exploration_weight=uct_exploration_weight, epsilon_robust=epsilon_robust)
    agents_list = [agent_mcts, agent_rmcts_batched]

    policy_dict = dict()
    P_dict = dict()
    for idx, agent in enumerate(agents_list):
        #########################################################
        # Train from scratch: set big eval range or one state
        #########################################################
        policy_est, v_est, v_est_batches, v_true_batches = run(agent, env, eval_range, n_experiments, save_path=save_path)
        # policy_dict[agent.agent_id] = policy_est
        policy_dict[agent.agent_id] = agent.current_dp_policy

        #########################################################
        # Load from the repo
        #########################################################
        path = os.path.join(load_path, env.env_id + '_epsilon_' + str(epsilon_robust) +
                            '_gamma_' + str(discount_factor), agent.agent_id)

        with open(os.path.join(path, "agents_P.pkl"), "rb") as f:
            agents_P = pickle.load(f)
            P_dict[agent.agent_id] = agents_P

        policy_dict[agent.agent_id] = np.load(os.path.join(path, 'policy_est.npy'))

    simulation_returns_mcts = list()
    simulation_returns_rmcts_batched = list()
    for i in range(n_simulation):

        ## Testing on P_min
        P_testing = P_dict[agent_rmcts_batched.agent_id]

        ## Testing on the orignal environment
        # P_testing = env.T

        ## Adding random vector 1
        # ksi = np.random.uniform(-epsilon_robust, epsilon_robust, size=env.T.shape)
        # # Do not touch transition for terminal states
        # ksi[env.terminal_states] = 0
        # P_testing = env.T + ksi
        # P_testing = (P_testing.squeeze() - P_testing.min(axis=3)) / (P_testing.max(axis=3) - P_testing.min(axis=3))
        # P_testing = P_testing / np.expand_dims(P_testing.sum(axis=2), axis=2)
        # # Shape: nS x nA x NT x nS
        # P_testing = np.expand_dims(P_testing, axis=2)
        # P_testing = 0.8 * env.T + 0.2 * P_testing

        ## TESTING
        # delta_2 = abs(env.T - P_testing).sum(axis=3)
        # delta_2_mask = delta_2 > epsilon_robust
        # delta_2_count.append(delta_2_mask.sum())
        # print(delta_2_mask.sum())

        ## Adding random vector 2
        # ksi = np.random.uniform(0, 1, size=env.T.shape)
        # ksi = ksi / np.expand_dims(ksi.sum(axis=3), axis=2)
        # assert ksi.sum() == 96
        # assert (ksi[0, 0, 0].sum() > 0.9999) and (ksi[0, 0, 0].sum() <= 1.00009), ksi[0, 0, 0].sum()
        # ## Do not touch transition for terminal states
        # ksi[env.terminal_states] = env.T[env.terminal_states]
        # lam = np.random.uniform(0, 0.6)
        # P_testing = (1 - lam) * env.T + lam * ksi

        # See what happens when you employ non-robust policy in worst-case model
        env_P_min = deepcopy(env)
        env_P_min.T = P_testing
        env_P_min.create_transition_dict_from_T()
        env.reset(start_state=eval_range[0])

        n_games = 1

        n_successes, returns_mcts = play_game(policy_dict[agent_mcts.agent_id], env_P_min,
                                              discount_factor=discount_factor, n_games=n_games,
                                              starting_state=eval_range[0])
        simulation_returns_mcts.append(returns_mcts)

        n_successes, returns_rmcts_batched = play_game(policy_dict[agent_rmcts_batched.agent_id], env_P_min,
                                                       discount_factor=discount_factor, n_games=n_games,
                                                       starting_state=eval_range[0])
        simulation_returns_rmcts_batched.append(returns_rmcts_batched)

    fig = plt.figure(figsize=(10, 6))

    simulation_returns_mcts = np.array(simulation_returns_mcts)
    simulation_returns_rmcts_batched = np.array(simulation_returns_rmcts_batched)

    plt.hist(simulation_returns_rmcts_batched.mean(axis=1), label='robust', color='green')
    plt.hist(simulation_returns_mcts.mean(axis=1), label="non-robust", alpha=0.5, color='red')
    plt.ylabel("Frequency")
    plt.xlabel("Discounted Return")
    plt.title(f"{env.env_id}_disc_return_freqs_under_")
    plt.legend()
    fig.show()

    print("Average MCTS return and its std ", np.mean(simulation_returns_mcts, axis=1).mean(),
          np.mean(simulation_returns_mcts, axis=1).std())
    print("Average rMCTS-Batched return and its std ", np.mean(simulation_returns_rmcts_batched, axis=1).mean(),
          np.mean(simulation_returns_rmcts_batched, axis=1).std())
