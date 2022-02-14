import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from simulators.battery import BatterySystem
from experience_replay.exp_replay import ReplayBuffer, PrioritizedReplayBuffer
from agents.agent_rule_based import AgentRB


# Specs for the battery
BAT_STOR = 0.0
BAT_CAP = 1.0
SoC_MIN = 0.0
SoC_MAX = 5.0
EFF_LOSS_AC_DC = 1.0
DEGRAD_PER_CYCLE = 0
MAX_CYCLES_DAY = 100


def add_transitions_to_buffer(transitions, buffer):
    if type(buffer) == ReplayBuffer or type(buffer) == PrioritizedReplayBuffer:
        for (f_t, a, r, f_tp1, done) in transitions:
            obs_t = np.hstack((f_t))
            obs_tp1 = np.hstack((f_tp1))
            buffer.add(obs_t, a, r, obs_tp1, done)


def test(agent, env, n_tests=1, iteration="NA"):
    actions_perfomed = []
    rewards_gotten = []

    for i in range(n_tests):
        env = BatterySystem(
            BAT_STOR,
            BAT_CAP,
            SoC_MIN,
            SoC_MAX,
            EFF_LOSS_AC_DC,
            DEGRAD_PER_CYCLE,
            MAX_CYCLES_DAY,
            historic_prices_period=("2018-01-25 00:00:00", "2018-02-28 00:00:00"),
        )
        obs_t = env.reset(episode_len_ptu=96 * 4)
        t = 0
        # Needed for Rule-based policies
        agent.env = env

        # ======= Testing/Debug =======
        print(f"Evaluation at {iteration}")
        print("Starting state: ", obs_t)
        # =============================

        while True:
            action = agent.select_action(state=np.hstack((obs_t)), dt_state=env.datetime_current, epsilon=0.0)
            # action = agent.select_action(obs_t, epsilon=0.0)
            t += 1
            obs_tp1, reward, done, _ = env.step(action)
            ## ======= Testing/Debug =======
            # print(env.charge_amount)
            ## =============================
            actions_perfomed.append(action)
            rewards_gotten.append(reward)
            if done:
                break
            obs_t = obs_tp1

        # ====================== Plotting ============================
        # For plotting in test() function
    fig = plt.figure(constrained_layout=True, figsize=(15, 7))
    gs1 = fig.add_gridspec(nrows=15, ncols=3)
    ax1 = fig.add_subplot(gs1[4:-4, :])
    ax2 = fig.add_subplot(gs1[-4:, :], sharex=ax1)
    ax0 = fig.add_subplot(gs1[1:4, :], sharex=ax1)

    # Set axis and titles
    fig.suptitle(f'Evaluation at iteration {iteration}', fontsize=16)
    ax0.set_title("Battery state of charge")
    ax0.set_ylabel("% Charged")
    ax1.set_title("Market Price")
    ax1.set_ylabel("EUR/MWh")
    ax2.set_title("Accumulated Revenue")
    ax2.set_ylabel("EUR")

    ax1.plot(env.df_episode.afnemen, label='afnemen')
    ax1.plot(env.df_episode.invoeden, label='invoden')
    ax1.scatter(
        env.df_episode[env.df_episode.action == 1].index,
        env.df_episode[env.df_episode.action == 1]["afnemen"],
        marker="x",
        color="r",
        s=30,
        label="charged",
    )
    ax1.scatter(
        env.df_episode[env.df_episode.action == 0].index,
        env.df_episode[env.df_episode.action == 0]["invoeden"],
        marker="o",
        color="g",
        s=30,
        label="discharged",
    )
    ax1.grid(True)
    ax1.legend(fancybox=True, framealpha=0.01)

    # Forward fill because we might be stepping with 15 minutes
    ax0.plot(env.df_episode.SoC.ffill())
    ax0.set_yticks(list(range(0, int(env.SoC_max) + 1)))
    ax0.set_yticklabels(list(range(0, int(env.SoC_max) + 1)), )
    ax0.grid(True)

    # Evaluate the agent in money terms
    charge_delta = env.df_episode.SoC.shift(-15) - env.df_episode.SoC
    # We charged
    pos_charge_delta = charge_delta[charge_delta > 0]
    # We discharged
    neg_charge_delta = charge_delta[charge_delta < 0]
    # We spent on charging
    costs_charge = -pos_charge_delta * env.df_episode.loc[pos_charge_delta.index].afnemen
    # Negative delta is we discharged thus should be converted to revenue
    profits_discharge = -neg_charge_delta * env.df_episode.loc[neg_charge_delta.index].invoeden
    profits_discharge.name = 'profit_loss'
    costs_charge.name = 'profit_loss'
    df = pd.DataFrame(index=env.df_episode.index)
    df.loc[costs_charge.index, 'profit_loss'] = costs_charge
    df.loc[profits_discharge.index, 'profit_loss'] = profits_discharge
    df['cum_sum'] = df.profit_loss.cumsum()
    ax2.plot(df.cum_sum.ffill().fillna(0))
    ax2.grid(True)

    ## Plot vertical lines
    # for i, x in enumerate(env.df_episode[env.df_episode.afnemen < 0].index):
    #     if x in env.df_episode[env.df_episode.action == 1].index:
    #         ax1.axvline(x=x, ymin=-0, ymax=10, c="red", linewidth=2, zorder=0, clip_on=False, alpha=0.1)
    #         ax2.axvline(x=x, ymin=0, ymax=10, c="red", linewidth=2, zorder=0, clip_on=False, alpha=0.1)

    fig.tight_layout()
    plt.show()

    print("Ending state: ", obs_t)
    print("rewards gotten:    ", rewards_gotten)
    print("Sum of the rewards: ", sum(rewards_gotten))
    print("actions performed: ", actions_perfomed)
    print()
    # =================================================

    return actions_perfomed


def main(params):
    np.random.seed(params["seed"])
    buffer = ReplayBuffer(params["buffer_size"])

    env = BatterySystem(
        BAT_STOR,
        BAT_CAP,
        SoC_MIN,
        SoC_MAX,
        EFF_LOSS_AC_DC,
        DEGRAD_PER_CYCLE,
        MAX_CYCLES_DAY,
        historic_prices_period=("2018-01-01 00:00:00", "2018-03-01 00:00:00"),
    )
    env.reset()

    agent = params["agent"](env, buffer)

    train_episode = 0
    while train_episode < params["train_episodes"]:
        obs_t = env.reset(current_energy_stored=env.bat_stored, datetime_dt=env.datetime_current, episode_len_ptu=96*1)
        agent = params["agent"](env, buffer)

        while True:
            action = agent.select_action(
                np.hstack((obs_t)), dt_state=env.datetime_current
            )
            obs_tp1, reward, done, _ = env.step(action)

            if done:
                train_episode += 1
                break

            obs_t = obs_tp1

        if train_episode % params["test_every"] == 0:
            test(agent, env, n_tests=1, iteration=train_episode)

    env.close()

    # TODO: create profit evaluation
    returns, losses, env = 0, 0, env

    return returns, losses, env


def plot_results(returns, losses, fig_title="figure", color="orange"):
    y_returns = moving_average(returns, 20)
    plt.plot(range(len(y_returns)), y_returns, color=color, label="Returns")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Returns")
    plt.title(fig_title)
    plt.show()

    y_losses = moving_average(losses, 20)
    plt.plot(range(len(y_losses)), y_losses, label="Losses")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Losses")
    plt.title(fig_title)
    plt.show()


if __name__ == "__main__":
    n = 10
    parameters = {
        "buffer": ReplayBuffer,
        "buffer_size": 10,
        "agent": AgentRB,
        "batch_size": 32,
        "lr": 1e-3,
        "gamma": 0.2,
        "epsilon_delta_end": 0.01,
        "epsilon_min": 0.05,
        "target_network_interval": 100,
        "environment": "OCOD",
        "train_episodes": 1000,
        "test_every": 5,
        "seed": None,
    }

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    returns, losses, env = main(parameters)
    plot_results(returns, losses, fig_title=env.env_id, color="orange")

    # ========= DEBUG =========
    # =========================

    ## ========== Testing ==============
    ## =================================
