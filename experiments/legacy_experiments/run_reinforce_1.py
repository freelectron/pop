import torch
from torch.optim import Adam
from torch.nn.modules import MSELoss
import numpy as np
from experience_replay.exp_replay import ReplayBuffer
from agents.REINFORCE import Simple_PG
import matplotlib.pyplot as plt
from simulators.ev_smart_charging.ocod import OCOD


def update(algorithm, buffer, params, train_steps):
    """
    Update policy parameters.
    """
    # I cannot train REINFORCE if we are not finished episode, thus pass this only if we finished an episode
    obses_t, actions, log_prob_actions, rewards, obses_tp1, dones = list(zip(*buffer))
    # We got log_prob_actions from the model. they are already tensors
    # log_prob_actions = np.array(torch.stack(log_prob_actions))
    obses_t, actions, rewards, obses_tp1, dones = list(
        map(np.array, [obses_t, actions, rewards, obses_tp1, dones])
    )
    loss = algorithm.train(
        obses_t, actions, log_prob_actions, rewards, obses_tp1, dones
    )

    return loss


def add_transitions_to_buffer(transitions, buffer):
    if type(buffer) == ReplayBuffer:
        for (f_t, a, log_probs, r, f_tp1, done) in transitions:
            obs_t = np.hstack((f_t))
            obs_tp1 = np.hstack((f_tp1))
            buffer.add(obs_t, a, r, obs_tp1, done)


def test(algorithm, env, n_tests=1):
    actions_perfomed = []
    rewards_gotten = []
    for i in range(n_tests):
        # Check which Environment we are using
        if isinstance(env, OCOD):
            obs_t, optimal_assignment = env.reset()
        t = 0

        # ======= Testing/Debug =======
        print("Starting state: ", obs_t)
        print("Price function: ", env.price_function)
        print("Optimal hours to charge: ", env.optimal_ch_hours)
        # =============================

        while True:
            action = algorithm.predict(np.hstack((obs_t)), eval=True)
            t += 1
            if isinstance(env, OCOD):
                obs_tp1, reward, done = env.step(action)
                actions_perfomed.append(action)
                rewards_gotten.append(reward)
            if done:
                break

            obs_t = obs_tp1

    # ======= Testing/Debug =======
    print("rewards gotten:    ", rewards_gotten)
    print("actions performed: ", actions_perfomed)
    print("optimal actions:   ", optimal_assignment)
    print()
    # ============================

    return actions_perfomed == optimal_assignment


def main(params):
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    if params["environment"] == "OCOD":
        env = OCOD()
        s, _ = env.reset()

    state_shape = s.shape[0]

    #  Keep the buffer also for MC (further add weights)
    if params["buffer"] == ReplayBuffer:
        buffer = ReplayBuffer(params["buffer_size"])
        loss_function = params["loss_function"]()
    else:
        raise ValueError("Buffer type not found.")

    if params["agent"] == Simple_PG:
        algorithm = Simple_PG(
            state_shape,
            env.action_space.n,
            loss_function=loss_function,
            optimizer=params["optimizer"],
            lr=params["lr"],
            gamma=params["gamma"],
            epsilon_delta=1 / (params["epsilon_delta_end"] * params["train_steps"]),
            epsilon_min=params["epsilon_min"],
        )

    losses = []
    returns = []
    train_steps = 0
    episodes_length = []
    test_results = []

    print("Starting to train:", type(buffer))

    if isinstance(env, OCOD):
        obs_t, _ = env.reset()
    t = 0
    episode_loss = []
    episode_rewards = []
    episode_transitions = []
    episode_log_prob_actions = []
    while train_steps < params["train_steps"]:
        # env.render()
        if isinstance(algorithm, Simple_PG):
            action, log_probs = algorithm.predict(obs_t)
            # Because the size of batch is one. We only need to select Zero-th element
            log_prob_action = log_probs[0][action]
            # If we need to do importance weights we will need this
            episode_log_prob_actions.append(log_prob_action)
            t += 1
            if isinstance(env, OCOD):
                obs_tp1, reward, done = env.step(action)
                transition = (obs_t, action, log_prob_action, reward, obs_tp1, done)

            episode_transitions.append(transition)
            train_steps += 1

            if done:
                # Calculate late loss and update
                loss = update(algorithm, episode_transitions, params, train_steps)
                episode_loss.append(loss)
                episode_rewards.append(reward)

            if train_steps % params["test_every"] == 0:
                # Give the same setting and see if the agent learnt to act
                test_results.append(test(algorithm, env) - False)
                print(
                    f"Average return at timestep t={train_steps} for the last 1000 steps is {np.mean(returns[-50:]).mean()}"
                )
                print(
                    f"Average optimal assignment percentage for the last 1000 steps is {np.mean(test_results[-50:])}"
                )

        add_transitions_to_buffer(episode_transitions, buffer)
        losses.append(np.mean(episode_loss))
        returns.append(np.sum(episode_rewards))

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

    y_test_results = moving_average(test_results, 20)
    plt.plot(range(len(y_test_results)), y_test_results, label="Good actions %")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Good actions %")
    plt.title("One car One day")
    plt.show()
    # plt.savefig('Good_actions_percentage' + '.png')


if __name__ == "__main__":
    n = 10
    parameters = {
        "buffer": ReplayBuffer,
        "buffer_size": 128,
        "PER_alpha": 0.6,
        "PER_beta": 0.4,
        "agent": Simple_PG,
        "batch_size": 64,
        "hidden_size": (32,),
        "optimizer": Adam,
        "loss_function": MSELoss,
        "lr": 1e-3,
        "gamma": 1,
        "epsilon_delta_end": 0.75,
        "epsilon_min": 0.05,
        "target_network_interval": 50,
        "environment": "OCOD",
        "train_steps": 80000,
        "test_every": 100,
        "seed": 42,
    }

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    test_results, returns, losses = main(
        parameters
    )  # [main(parameters) for _ in range(n)]
    plot_results(test_results, returns, losses, fig_name="figure", color="orange")
