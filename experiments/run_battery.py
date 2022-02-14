import os
import logging
import pickle as pkl
import argparse
from datetime import datetime
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
from torch.optim import SGD, Adam
from torch.nn.modules import L1Loss, MSELoss
import torch

from simulators.battery import BatterySystem
from simulators.battery import DiscreteBatterySystem
from experience_replay.exp_replay import ReplayBuffer
from agents.agent_dqn import AgentDQN
from agents.agent_rdqn import AgentRDQN
from agents.agent_rule_based import AgentRB
from agents.agent_mcts import AgentMCTS


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


# Specs for the battery
BAT_STOR = 1.0
BAT_CAP = 1.0
SoC_MIN = 0.0
SoC_MAX = 1.0
EFF_LOSS_AC_DC = 1.0
DEGRAD_PER_CYCLE = 0
MAX_CYCLES_DAY = 50
# todo: now predictions for PTU1-6 is available from 2019-02-01 till 2019-10-01, make more data
DATA_RANGE_PERIOD = ("2019-02-01 00:00:00", "2019-10-01 00:00:00")
TRAIN_PERIOD = ("2019-02-01 00:00:00", "2019-02-01 00:02:00")
EVAL_PERIOD = ("2019-02-01 00:00:00", "2019-03-18 00:00:00")
EPISODE_LEN_PTU_TRAIN = 1  # 96 * 1
EPISODE_LEN_PTU_EVAL = 96  # 96 * 1


def plot_test_evaluation(env, iteration, test_run, rewards_gotten, rb_profit, rb_soc):
    """
    Plots graphs used to evaluate the agent on the test set.

    Args:
         iteration (int): at what stage in training are we
         rb_profit (pandas.Series): df wit a single column that hold accumulated profit for rule-based policies
    """
    # For plotting in test() function
    fig = plt.figure(constrained_layout=True, figsize=(20, 7))
    gs1 = fig.add_gridspec(nrows=15, ncols=3)
    ax1 = fig.add_subplot(gs1[4:-4, :])
    ax2 = fig.add_subplot(gs1[-4:, :], sharex=ax1)
    ax0 = fig.add_subplot(gs1[0:4, :], sharex=ax1)

    # Set axis
    ax0.set_title("Battery state of charge")
    ax0.set_ylabel("% Charged")
    ax1.set_title("Market Price")
    ax1.set_ylabel("EUR/MWh")
    ax2.set_title("Accumulated Revenue")
    ax2.set_ylabel("EUR")

    ax1.plot(env.df_episode.afnemen, label='afnemen')
    ax1.plot(env.df_episode.invoeden, label='invoden')
    ax1.plot(env.df_episode.invoeden_pred_price_pte0, label='invoeden_pred_price_pte0', alpha=.3)
    ax1.plot(env.df_episode.afnemen_pred_price_pte0, label='afnemen_pred_price_pte0', alpha=.3)
    # ax1.scatter(
    #     env.df_episode[env.df_episode.action == 1].index,
    #     env.df_episode[env.df_episode.action == 1]["afnemen"],
    #     marker="x",
    #     color="r",
    #     s=30,
    #     label="charged")
    # ax1.scatter(
    #     env.df_episode[env.df_episode.action == 0].index,
    #     env.df_episode[env.df_episode.action == 0]["invoeden"],
    #     marker="o",
    #     color="g",
    #     s=30,
    #     label="discharged")
    ax1.grid(True)
    ax1.legend(fancybox=True, framealpha=0.01)

    # Forward fill because we might be stepping with 15 minutes
    ax0.plot(env.df_episode.SoC.ffill(), label="Trained")
    ax0.plot(rb_soc.ffill(), label="Rule-based")
    ax0.set_yticks(list(range(0, int(env.SoC_max) + 1)))
    ax0.set_yticklabels(list(range(0, int(env.SoC_max) + 1)), )
    ax0.grid(True)
    ax0.legend(fancybox=True, framealpha=0.01)

    profit_loss = BatterySystem.calc_profit(env.df_episode, env.time_step_delta)
    ax2.plot(profit_loss, label='Trained')
    # For evaluation of rule-based
    if rb_profit is not None:
        ax2.plot(rb_profit, label='Rule-based')
    ax2.grid(True)
    ax2.legend(fancybox=True, framealpha=0.01)

    fig.suptitle(f'Evaluation run {test_run} | Iteration {iteration}', fontsize=16)

    episode_profit_str = f"Cumulated Revenue {profit_loss.iloc[-1]:.3f}"
    text(.1, 1.5, episode_profit_str, horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)

    if rewards_gotten:
        episode_profit_str = f"Episode's return {rewards_gotten:.3f}"
        text(.3, 1.5, episode_profit_str, horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)

    # Calculate how much load was steered
    charge_delta = env.df_episode.SoC.shift(-env.time_step_delta) - env.df_episode.SoC
    pos_charge_delta = charge_delta[charge_delta > 0]
    neg_charge_delta = charge_delta[charge_delta < 0]
    flexed = abs(pos_charge_delta.sum()) + abs(neg_charge_delta.sum())
    text(0.55, 1.5, f"Load steered [Trained]: {flexed}", horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)

    charge_delta = rb_soc.shift(-env.time_step_delta) - rb_soc
    pos_charge_delta = charge_delta[charge_delta > 0]
    neg_charge_delta = charge_delta[charge_delta < 0]
    flexed_rb = abs(pos_charge_delta.sum()) + abs(neg_charge_delta.sum())
    text(0.75, 1.5, f"Load steered [RB]: {flexed_rb}", horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)

    fig.tight_layout()
    # plt.savefig(SAVE_RESULTS_PATH+f"iter_{iteration:02d}_run_{test_run:02d}")
    plt.show()


def add_transitions_to_buffer(transitions, buffer):
    if type(buffer) == ReplayBuffer:
        for (f_t, a, r, f_tp1, done) in transitions:
            obs_t = np.hstack((f_t))
            obs_tp1 = np.hstack((f_tp1))
            buffer.add(obs_t, a, r, obs_tp1, done)


def test_rb(agent_rb, env, dt_eval_start, env_seed=25):
    """
    Evaluates the agent on the test set.

    Returns:
        actions_perfomed (list): performed actions during testing
        rb_agent_run (bool): whether to run rule-based agent in this call of `test()`
    """
    # To change seed from training time
    env.seed(env_seed)
    # Begin from the start of prespecified eval period
    datetime_dt = dt_eval_start
    obs_t = env.reset(datetime_dt=datetime_dt, episode_len_ptu=EPISODE_LEN_PTU_EVAL, test_time=True)
    # Needed for Rule-based policies
    agent_rb.env = env
    t = 0
    while True:
        action = agent_rb.select_action(state=np.hstack((obs_t)), dt_state=env.datetime_current, epsilon=0.0)
        # action = agent.select_action(obs_t, epsilon=0.0)
        t += 1
        obs_tp1, reward, done, _ = env.step(action)
        if done:
            break
        obs_t = obs_tp1

    return agent_rb.env.df_episode


def test(agent, env, dt_eval_start, n_tests=1, iteration="NA", plotting=True, rb_agent_run=False, env_seed=25):
    """
    Evaluates the agent on the test set.

    Returns:
        actions_perfomed (list): performed actions during testing
        rb_agent_run (bool): whether to run rule-based agent in this call of `test()`
    """
    # To change seed from training time
    env.seed(env_seed)
    env.create_tranistion_structure()
    # Begin from the start of prespecified eval period
    datetime_dt = dt_eval_start

    for run_idx in range(n_tests):
        # For rule-based evaluation
        rb_profit = None
        obs_t = env.reset(datetime_dt=datetime_dt, episode_len_ptu=EPISODE_LEN_PTU_EVAL, test_time=True)

        if rb_agent_run:
            # Needed for rule-based policies: agent has access to the dataframe with features
            agent.env = env
            agent_rb = AgentRB(copy(env), buffer=None)
            # Calculate profit
            df_episode_rb = test_rb(agent_rb, copy(env), dt_eval_start=datetime_dt, env_seed=env_seed)
            rb_profit = BatterySystem.calc_profit(agent_rb.env.df_episode, agent_rb.env.time_step_delta)
            rb_soc = df_episode_rb.SoC
        t = 0
        obs_t = [obs_t]

        # Warm-up period: collect sequence to do many-to-one prediction
        #                 you already have obs_t from reset so adjust [aka range(1, seq_length)]
        for seq_length in range(1, parameters['sequence_length']):   # agent.sequence_length):
            ########################
            env.generate_transition_matrix()
            env.generate_reward_matrix()
            # Collect sequence
            action, _ = agent.select_action(state=np.vstack((obs_t)),
                                         dt_state=env.datetime_current, epsilon=0.0,
                                         batch_size=1, sequence_length=seq_length,
                                         # Needed for MCTS
                                         done=False)
            ########################
            t += 1
            obs_tp1, reward, done, _ = env.step(action)
            obs_t.append(obs_tp1)

        # Fix sequence length
        seq_length = len(obs_t)
        # Store results
        actions_perfomed = list()
        rewards_gotten = list()

        while True:
            ########################
            env.generate_transition_matrix()
            env.generate_reward_matrix()
            action, _ = agent.select_action(state=np.vstack((obs_t)), dt_state=env.datetime_current, epsilon=0.0,
                                         batch_size=1, sequence_length=seq_length,
                                         # Needed for MCTS
                                         done=False)
            ########################
            obs_tp1, reward, done, _ = env.step(action)
            actions_perfomed.append(action)
            rewards_gotten.append(reward)
            if done:
                break
            t += 1
            # Keep a sequence of last states
            obs_t.append(obs_tp1)
            obs_t.pop(0)

        # Plot evaluation run
        if plotting:
            plot_test_evaluation(env, iteration, run_idx,
                                 rewards_gotten=sum(rewards_gotten), rb_profit=rb_profit, rb_soc=rb_soc)
        # Store timestamp for the next consequtive evalution
        datetime_dt = env.datetime_current

    # TODO: define conditions on when to save the model
    # torch.save(agent.model.state_dict(), SAVE_RESULTS_PATH+"best_model_dict")

    return agent.env.df_episode


def main(params):
    np.random.seed(params["seed"])
    # Select type of experience replay using the parameters
    if params["buffer"] == ReplayBuffer:
        buffer = ReplayBuffer(params["buffer_size"])
    else:
        raise ValueError("Buffer type not found.")

    env = DiscreteBatterySystem( #BatterySystem(
        BAT_STOR,
        BAT_CAP,
        SoC_MIN,
        SoC_MAX,
        EFF_LOSS_AC_DC,
        DEGRAD_PER_CYCLE,
        MAX_CYCLES_DAY,
        historic_prices_period=DATA_RANGE_PERIOD,
        run_period=TRAIN_PERIOD,
    )
    env.seed(params['seed'])
    env.reset()
    env.create_tranistion_structure()

    agent = params["agent"](
        env=env,
        buffer=buffer,

        # Action space need for MCTS
        action_space=env.action_space,
        max_depth=3,
        horizon=7,
        # Set high explorarion weight
        uct_exploration_weight=1,
        uct_tree_policy=True,
        discount_factor=.5,

        loss_function=params["loss_function"](),
        optimizer=params["optimizer"],
        lr=params["lr"],
        double_dqn=params['use_double_dqn'],
        gamma=params["gamma"],
        epsilon_delta=params["epsilon_delta_end"],
        epsilon_min=params["epsilon_min"],
        batch_size=params['batch_size'],
        sequence_length=params['sequence_length'],
        seed=params['seed'],
    )

    losses = []
    returns = []
    train_episode = 0
    iteration = 0
    lengths = []

    print(params["loss_function"])
    print(params["optimizer"])

    while train_episode < params["train_episodes"]:
        train_episode_losses = []
        train_episode_rewards = []
        episode_transitions = []
        episode_actions = []
        t = 0
        obs_t = env.reset(current_energy_stored=env.bat_stored, datetime_dt=env.datetime_current, episode_len_ptu=EPISODE_LEN_PTU_TRAIN)

        # Do mini-batch training if only not treed-based/random sampling technique
        if not isinstance(agent, AgentMCTS):
            while True:
                ########################
                action, _ = agent.select_action(state=np.vstack((obs_t)).reshape((1, -1)),
                                             dt_state=env.datetime_current, epsilon=agent.get_epsilon(train_episode))
                # action = agent.select_action(state=np.hstack((obs_t)), dt_state=env.datetime_current, epsilon=1.0)
                ########################
                t += 1
                iteration += 1
                obs_tp1, reward, done, _ = env.step(action)
                transition = (obs_t, action, reward, obs_tp1, done)

                episode_transitions.append(transition)
                train_episode_rewards.append(reward)
                episode_actions.append(action)

                if len(buffer) >= params["batch_size"]:
                    loss = agent.train(target_update=train_episode % params["target_network_interval"] == 0)
                    train_episode_losses.append(loss)

                if done:
                    lengths.append(t)
                    train_episode += 1
                    break

                obs_t = obs_tp1

        add_transitions_to_buffer(episode_transitions, buffer)
        losses.append(np.mean(train_episode_losses))
        returns.append(np.sum(train_episode_rewards))

        # =================
        # Logging
        # =================
        logger.info(f"iteration_{train_episode}:: Batch Loss [ sum {sum(train_episode_losses)} | "
                    f"mean {sum(train_episode_losses) / (len(train_episode_losses) or 1)} ]")
        logger.info(f"iteration_{train_episode}:: Learning   [ epsilon {agent.get_epsilon(train_episode)} ]")
        logger.info("\n")

        if train_episode % params["test_every"] == 0:
            test(agent, copy(env), dt_eval_start=datetime.strptime(EVAL_PERIOD[0], "%Y-%m-%d %H:%M:%S"),
                 n_tests=1, iteration=train_episode, rb_agent_run=True, env_seed=params['seed'])

    env.close()

    return returns, losses, env


def plot_results(returns, losses, fig_title="figure", color="orange"):
    y_returns = moving_average(returns, 20)

    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    plt.plot(range(len(y_returns)), y_returns, color=color, label="Returns")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Returns")
    plt.title(fig_title)
    plt.savefig(SAVE_RESULTS_PATH+"returns")
    # plt.show()

    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    y_losses = moving_average(losses, 20)
    plt.plot(range(len(y_losses)), y_losses, label="Losses")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Losses")
    plt.title(fig_title)

    plt.savefig(SAVE_RESULTS_PATH+"losses")
    # plt.show()


if __name__ == "__main__":

    def get_argparse():
        """Needed for sphinx.ext.napoleon"""
        parser = argparse.ArgumentParser(
            description="""
                give args.
        """,
            formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument(
            "--save-results-path",
            type=str,
            default="./storage/results/run_0/",
            help="Path where to save results}")

        parser.add_argument(
            "--use-sgd",
            type=bool,
            default=False,
            help="Use SGD,otherwise Adam",
        )

        parser.add_argument(
            "--use-ls1",
            type=bool,
            default=False,
            help="Use L1 loss, otherwise MSE",
        )

        parser.add_argument(
            "--use-double-dqn",
            type=bool,
            default=False,
            help="Use Double Deep Q-Network, otherwise not, and be scare of Maximization Bias.",
        )

        parser.add_argument(
            "--sequence-length",
            type=int,
            default=1,
            help="If you want to train an LSTM set this to something other than 1.",
        )
        return parser

    parser = get_argparse()

    args = parser.parse_args()

    # ARGS for training
    SAVE_RESULTS_PATH = args.save_results_path
    USE_SGD = args.use_sgd
    USE_LS1 = args.use_ls1
    USE_DOUBLE_DQN = True  # args.use_double_dqn
    SEQUENCE_LENGTH = 3    # args.sequence_length

    # Make save directory if does not exist
    if not os.path.exists(SAVE_RESULTS_PATH):
        os.makedirs(SAVE_RESULTS_PATH)

    n = 10
    parameters = {
        "buffer": ReplayBuffer,
        "buffer_size": 4096,
        "sequence_length": SEQUENCE_LENGTH,
        "agent": AgentMCTS, #AgentRDQN if SEQUENCE_LENGTH > 1 else AgentDQN,
        "batch_size": 2,
        "use_double_dqn": USE_DOUBLE_DQN,
        "optimizer": Adam if not USE_SGD else SGD,
        "loss_function": MSELoss if not USE_LS1 else L1Loss,
        "lr": 1e-4 if not USE_SGD else 1e-3,
        "gamma": 0.8,
        "epsilon_delta_end": 0.01,
        "epsilon_min": 0.05,
        "target_network_interval": 1,
        "train_episodes": 2,
        "test_every": 1,
        "seed": 25,
    }
    # Serialize parameters into a file:
    with open(SAVE_RESULTS_PATH+'params.pkl', 'wb') as fp:
        pkl.dump(parameters, fp)

    print(parameters)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    returns, losses, env = main(parameters)

    plot_results(returns, losses, fig_title=env.env_id, color="orange")
