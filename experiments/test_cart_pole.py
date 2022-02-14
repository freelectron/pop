import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam
from torch.nn.modules import L1Loss, MSELoss
import gym

from experience_replay.exp_replay import ReplayBuffer, PrioritizedReplayBuffer
from agents.agent_dqn import AgentDQN
from agents.agent_rdqn import AgentRDQN


def add_transitions_to_buffer(transitions, buffer):
    if type(buffer) == ReplayBuffer or type(buffer) == PrioritizedReplayBuffer:
        for (f_t, a, r, f_tp1, done) in transitions:
            obs_t = np.hstack((f_t))
            obs_tp1 = np.hstack((f_tp1))
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

    env = gym.make("CartPole-v0")
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
        batch_size=params['batch_size'],
        sequence_length=params['sequence_length'],
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
        obs_t = env.reset()

        while True:
            epsilon = (params['train_episodes'] - train_episode+100)/params['train_episodes']
            epsilon = epsilon if epsilon > 0.05 else 0.1

            action = agent.select_action(np.hstack((obs_t)), epsilon=epsilon)


            # print(action)


            t += 1
            iteration += 1
            obs_tp1, reward, done, _ = env.step(action)
            transition = (obs_t, action, reward, obs_tp1, done)

            episode_transitions.append(transition)
            episode_rewards.append(reward)
            episode_actions.append(action)

            if len(buffer) >= params["batch_size"] * params['sequence_length']:
                loss = agent.train(batch_size=params["batch_size"],sequence_length=params['sequence_length'])
                episode_loss.append(loss)

            if done == True:
                lengths.append(t)
                train_episode += 1
                break

            obs_t = obs_tp1

        add_transitions_to_buffer(episode_transitions, buffer)
        losses.append(np.mean(episode_loss))
        returns.append(np.sum(episode_rewards))

        if train_episode % params["test_every"] == 0:
            # ========= Test =========
            obs_t = env.reset()
            while True:
                action = agent.select_action(np.hstack((obs_t)), epsilon=0.0)
                t += 1
                obs_tp1, reward, done, _ = env.step(action)

                if done == True:
                    print("test episode length", t)
                    break
                obs_t = obs_tp1
            print("Agent epsion: ", agent.epsilon)
            # =========================

    env.close()

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
        "agent": AgentRDQN,  # AgentDQN,  #
        "batch_size": 5,
        "optimizer": Adam,
        "loss_function": MSELoss,
        "lr": 1e-3,
        "gamma": 0.8,
        "target_network_interval": 100,
        "environment": "OCOD",
        "train_episodes": 300,
        "test_every": 10,
        "sequence_length": 1,

        "epsilon_delta_end": 0.01,
        "epsilon_min": 0.05,
        "PER_alpha": 0.6,
        "PER_beta": 0.4,
        "seed": 42,
        "alpha": 0.6,
        "beta": 0.4,
    }

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    test_results, returns, losses = main(parameters)
    plot_results(test_results, returns, losses, fig_name="figure", color="orange")

    # ========= DEBUG =========
    # =========================
