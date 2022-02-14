import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.nn.modules import L1Loss
import gym

from experience_replay.exp_replay import ReplayBuffer, PrioritizedReplayBuffer
from agents.agent_dqn import AgentDQN
from agents.agent_qlearning import AgentQLearning


def plot_gridworld_value(Q, env, policy):
    V = [np.max(Q[i]) for i in range(0, env.nS)]
    V = np.array(V).reshape(env.shape)
    plt.figure()
    c = plt.pcolormesh(V, cmap="gray")  # "BuPu")
    for y in range(V.shape[0]):
        for x in range(V.shape[1]):
            plt.text(
                x + 0.5,
                y + 0.5,
                "%.2f" % V[y, x],
                horizontalalignment="center",
                verticalalignment="center",
                color="red",
                alpha=0.7,
            )

    plt.title(f"Checking Q values.")
    plt.colorbar(c)
    plt.gca().invert_yaxis()  # In the array, first row = 0 is on top
    plt.show()


def add_transitions_to_buffer(transitions, buffer):
    if type(buffer) == ReplayBuffer or type(buffer) == PrioritizedReplayBuffer:
        for (f_t, a, r, f_tp1, done) in transitions:
            obs_t = f_t
            obs_tp1 = f_tp1
            buffer.add(obs_t, a, r, obs_tp1, done)


def main(params):
    np.random.seed(params["seed"])
    # Select type of experience replay using the parameters
    if params["buffer"] == ReplayBuffer:
        buffer = ReplayBuffer(params["buffer_size"])
        loss_function = params["loss_function"]()
    elif params["buffer"] == PrioritizedReplayBuffer:
        buffer = PrioritizedReplayBuffer(
            params["buffer_size"], params["PER_alpha"], params["PER_beta"]
        )
        loss_function = params["loss_function"](reduction="none")
    else:
        raise ValueError("Buffer type not found.")

    env = gym.make("CliffWalking-v0")

    # Call reset to get a proper state space shape
    env.reset()

    agent = params["agent"](
        env,
        buffer,
        loss_function=loss_function,
        optimizer=params["optimizer"],
        lr=params["lr"],
        gamma=params["gamma"],
        epsilon_delta=params["epsilon_delta_end"],
        epsilon_min=params["epsilon_min"],
    )
    losses = []
    returns = []
    train_episode = 0
    iteration = 0
    lengths = []
    test_results = []

    while train_episode < params["train_episodes"]:
        episode_loss = []
        episode_rewards = []
        episode_transitions = []
        episode_actions = []
        t = 0

        # For windy gridworld, exploration start is not working. Not sure about ClifWalking
        obs_t = env.reset()
        obs_t = int(np.random.rand(1)[0] * (env.nS - 1))
        env.s = obs_t
        print(obs_t)

        while True:
            # print(agent.get_epsilon(iteration))
            # TODO: working with experience replay in tabular setting is tricky (windy gridworld is still not solved)
            action = agent.select_action(
                obs_t, epsilon=0.9
            )  # agent.get_epsilon(iteration))
            t += 1
            iteration += 1
            obs_tp1, reward, done, _ = env.step(action)
            transition = (obs_t, action, reward, obs_tp1, done)
            episode_transitions.append(transition)
            episode_rewards.append(reward)
            episode_actions.append(action)

            if len(buffer) >= params["batch_size"]:

                # print(len(buffer))

                # loss = agent.train(params["batch_size"])
                policy, loss = agent.train_model_based(env=env)
                episode_loss.append(loss)

            if done:
                lengths.append(t)
                train_episode += 1
                break

            # # ========= Test =========
            # print("Observation", obs_t)
            # print(agent.get_epsilon(train_episode))
            # obs_t = env.render()
            # # =========================

            obs_t = obs_tp1

        add_transitions_to_buffer(episode_transitions, buffer)
        losses.append(np.mean(episode_loss))
        returns.append(np.sum(episode_rewards))

        if train_episode % params["test_every"] == 0:
            t = 0
            # ========= Test =========
            plot_gridworld_value(Q=agent.Q, env=env, policy=0)
            obs_t = env.reset()
            # print(env.render())

            while True:
                action = agent.select_action(obs_t, epsilon=0.0)
                t += 1
                obs_tp1, reward, done, _ = env.step(action)

                # print(env.render())

                if done:
                    print("test episode length", t)
                    break

                obs_t = obs_tp1
            print("Agent epsion: ", agent.epsilon)
            # =========================

    # env.close()

    return test_results, returns, losses


def plot_results(test_results, returns, losses, fig_name="figure", color="orange"):
    y_returns = moving_average(returns, 20)
    plt.plot(range(len(y_returns)), y_returns, color=color, label="Returns")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Returns")
    plt.title("One car One day")
    plt.show()
    # plt.savefig('Returns' + '.png')

    y_losses = moving_average(losses, 20)
    plt.plot(range(len(y_losses)), y_losses, label="Losses")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Losses")
    plt.title("One car One day")
    # plt.savefig('Losses' + '.png')
    plt.show()

    try:
        y_test_results = moving_average(test_results, 20)
        plt.plot(range(len(y_test_results)), y_test_results, label="Good actions %")
        plt.legend()
        plt.xlabel("Training steps")
        plt.ylabel("Good actions %")
        plt.title("One car One day")
        plt.show()
    except:
        print("Could not plot.")
    # plt.savefig('Good_actions_percentage' + '.png')


if __name__ == "__main__":
    n = 10
    parameters = {
        "buffer": ReplayBuffer,
        "buffer_size": 10000,
        "PER_alpha": 0.6,
        "PER_beta": 0.4,
        "agent": AgentQLearning,
        "batch_size": 64,
        "optimizer": SGD,
        "loss_function": L1Loss,
        "lr": 1e-3,
        "gamma": 0.8,
        "epsilon_delta_end": 0.01,
        "epsilon_min": 0.05,
        "target_network_interval": 100,
        "environment": "OCOD",
        "train_episodes": 100,
        "test_every": 10,
        "seed": None,
        "alpha": 0.6,
        "beta": 0.4,
    }

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    test_results, returns, losses = main(parameters)
    plot_results(test_results, returns, losses, fig_name="figure", color="orange")

    # ========= DEBUG =========
    # =========================
