import os
import logging
from collections import defaultdict
import time

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


def run(agent, env, eval_range, n_experiments, save_path=None, load_path=None):
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

    if load_path is None:
        # Get the policy and value function
        start = time.time()
        # Create value function with mcts procedure
        policy_est, v_est, v_est_batches, v_true_batches = agent.evaluate(n_experiments, eval_range=eval_range)
        end = time.time()
        logging.info(
            f"Finished evaluating {len(eval_range)} state(s) in {(end - start) // 60} mins {(end - start) % 60} secs")
        if save_path:
            # MCTS estimates
            np.save(
                os.path.join(save_path, env.env_id + "_" + agent.agent_id + "_tensor_mcts_vals__eval_batches_" + str(
                    agent.n_batches) +
                             "_batch_size" + str(agent.n_evaluation) + "_robust_eps_" + str(agent.epsilon_robust)),
                v_est_batches)
            # Value Iteration solution
            np.save(os.path.join(save_path, env.env_id + "_" + agent.agent_id + "_tensor_dp_vals__eval_batches_" + str(
                agent.n_batches) +
                                 "_batch_size" + str(agent.n_evaluation) + "_robust_eps_" + str(agent.epsilon_robust)),
                    v_true_batches)

    else:
        v_est_batches = np.load(os.path.join(load_path,
                                             env.env_id + "_" + agent.agent_id + "_tensor_mcts_vals__eval_batches_" + str(
                                                 agent.n_batches) +
                                             "_batch_size" + str(agent.n_evaluation) + "_robust_eps_" + str(
                                                 agent.epsilon_robust) + ".npy"))
        v_true_batches = np.load(os.path.join(load_path,
                                              env.env_id + "_" + agent.agent_id + "_tensor_dp_vals__eval_batches_" + str(
                                                  agent.n_batches) +
                                              "_batch_size" + str(agent.n_evaluation) + "_robust_eps_" + str(
                                                  agent.epsilon_robust) + ".npy"))
        policy_est = np.zeros([v_est_batches.shape[0], env.nS, env.nA])
        v_est = np.zeros([v_est_batches.shape[0], env.nS])

    return policy_est, v_est, v_est_batches, v_true_batches


if __name__ == "__main__":

    SEED = 20
    # Which states you want to evaluate
    EVAL_RANGE = [10]  # + [1, 5] + list(range(7, 30)) + [33, 34]
    DISCOUNT_FACTOR = 0.95
    NUM_EXPERIMENTS = 1
    SAVE_PATH = "storage/results_report"

    logging.basicConfig(filename='example.log', level=logging.DEBUG)

    # TODO: introduce a config file or pass as cmd args
    n_experiments = NUM_EXPERIMENTS
    disount_factor = DISCOUNT_FACTOR
    eval_range = EVAL_RANGE
    save_path = SAVE_PATH
    num_rollouts = 200000
    # Evaluate after how many simulations
    n_evaluation = 40000
    num_eval_batches = num_rollouts // n_evaluation
    epsilon_robust = 0.2
    uct_exploration_weight = 5

    # What environment to use
    env = FrozenLakeEnv()  # LavaWorld()  #

    # Define what you need to do modelling
    agent_mcts = AgentMCTS(env, num_rollouts=num_rollouts, n_evaluation=n_evaluation,
                           discount_factor=disount_factor, uct_tree_policy=True, horizon=20,
                           uct_exploration_weight=uct_exploration_weight)
    # Batched version starts from
    agent_rmcts_batched = AgentRMCTSBatched(env, num_rollouts=num_rollouts, num_batches=num_eval_batches,
                                            discount_factor=disount_factor, uct_tree_policy=True, horizon=20,
                                            uct_exploration_weight=uct_exploration_weight, epsilon_robust=epsilon_robust)
    agent_rmcts_iter = AgentRMCTSIterative(env, num_rollouts=num_rollouts, n_evaluation=n_evaluation,
                                           discount_factor=disount_factor, uct_tree_policy=True, horizon=20,
                                           uct_exploration_weight=uct_exploration_weight, epsilon_robust=epsilon_robust)
    agents_list = [agent_mcts, agent_rmcts_batched, agent_rmcts_iter]  #

    # For nice legends and axis
    plotting.set_report_style()
    # Select a value to plot from eval_range
    eval_state = eval_range[0]

    ## Set-up for plotting
    fig_convergence_plot = plt.figure(figsize=(10, 6))
    axs = {}

    for idx, agent in enumerate(agents_list):
        if idx == 2:
            n_experiments = 1

        # Run an experiment
        policy_est, v_est, v_est_batches, v_true_batches = run(agent, env, eval_range, n_experiments, load_path=save_path)

        # Plot the average values and policies (0th dimension is number of MCTS runs)
        average_policy = policy_est.sum(axis=0) / n_experiments
        v_roots = v_est.mean(axis=0)
        # plot_gridworld_value(v_roots.reshape(env.shape), env, average_policy, save_path)

        # Plot the convergence of the value for the root state
        x = np.array([agent.n_evaluation * i for i in range(0, agent.n_batches + 1)])

        # Mean value
        y_est = np.array([0] + [v_est_batches[:, i, eval_state].mean() for i in range(agent.n_batches)])
        y_true = np.array([0] + [v_true_batches[:, i, eval_state].mean() for i in range(agent.n_batches)])
        # Standard deviations
        y_est_std = np.array([0] + [v_est_batches[:, i, eval_state].std() for i in range(agent.n_batches)])

        if idx != 1:
            y_true_std = np.array([0] + [v_true_batches[:, i, eval_state].std() for i in range(agent.n_batches)])

        axs[idx] = fig_convergence_plot.add_subplot(111)

        axs[idx].plot(x, y_est, color=agent.agent_colour, label=f"{agent.agent_id}")
        axs[idx].fill_between(x, y_est - y_est_std, y_est + y_est_std, color=agent.agent_colour, alpha=0.2)

        if idx != 1:
            axs[idx].plot(x, y_true, color=agent.agent_colour, label='VI' if agent.agent_id == "MCTS" else 'rVI', alpha=0.5)
            axs[idx].fill_between(x, y_true - y_true_std, y_true + y_true_std, color=agent.agent_colour, alpha=0.2)

        average_policy = policy_est.sum(axis=0) / n_experiments
        v_roots = v_roots.mean(axis=0)

    axs[0].set_xlabel('Simulations')
    axs[0].set_ylabel(f'Value s_{eval_state}')
    axs[0].legend(bbox_to_anchor=(0.97, 0.97), loc='upper left')
    fig_convergence_plot.show()
