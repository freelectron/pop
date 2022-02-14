import random

import numpy as np
import torch.nn as nn
from torch import optim
import torch
import matplotlib.pyplot as plt

# from simulators.ev_smart_charging.ocod import OCOD


class PolicyNetwork(nn.Module):
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(8, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


def select_action(model, state):
    # Samples an action according to the probability distribution induced by the model
    # Also returns the log_probability
    # YOUR CODE HERE
    log_p = model(torch.tensor(state).float())
    action = torch.multinomial(torch.exp(log_p), 1).item()
    return action, log_p[action]


def compute_reinforce_loss(episode, discount_factor):
    states, log_probs, rewards = zip(*episode)
    log_probs = torch.stack(log_probs)
    episode_return = 0.0
    returns = []
    for r in rewards[::-1]:
        episode_return = r + discount_factor * episode_return
        returns.append(episode_return)

    returns = torch.tensor(returns[::-1])
    returns -= torch.mean(returns)
    returns /= torch.std(returns)

    loss = -(returns * log_probs).sum()
    return loss


def run_episode(env, model):
    episode = []
    done = False
    state, _ = env.reset()
    single_transition = []
    while not done:
        previous_state = state
        action, log_p = select_action(model, state)
        state, reward, done = env.step(action)
        episode.append((previous_state, log_p, reward))
    return episode


def run_episodes_policy_gradient(model, env, num_episodes, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    success_list = []
    episode_durations = []
    for i in range(num_episodes):
        episode = run_episode(env, model)
        loss = compute_reinforce_loss(episode, discount_factor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        states, log_probs, actions = list(zip(*episode))
        success_list.append(env.perfect_actions == list(actions))
        if i % 100 == 0:
            print("Episode: ", i)
            print("Optimal actions: ", env.perfect_actions)
            print("Optimal actions: ", list(actions))
            print("Print loss: ", loss.item())
            print(
                "Average optimal assignment percentage for the last 50 steps is ",
                np.mean(success_list[-50:]),
            )

        episode_durations.append(env.perfect_actions == list(actions))

    return episode_durations


if __name__ == "__main__":
    num_episodes = 10000
    discount_factor = 0.99
    learn_rate = 0.001
    seed = 42
    num_hidden = 128
    random.seed(seed)
    torch.manual_seed(seed)

    env = OCOD()
    s, _ = env.reset()

    model = PolicyNetwork(num_hidden)

    episode_durations_policy_gradient = run_episodes_policy_gradient(
        model, env, num_episodes, discount_factor, learn_rate
    )

    plt.plot(episode_durations_policy_gradient)
    plt.title("Episode optimal actions")
    plt.legend(["Policy gradient"])
    plt.show()
