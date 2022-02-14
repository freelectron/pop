import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.modules import MSELoss
from torch.optim import SGD


class PolicyNetwork(nn.Module):
    def __init__(self, num_input, num_output, num_hidden=(64,)):
        super().__init__()
        layers = []
        # This will be something like this: (8=state space dims) (8, 64, 32, 16, 2) (2=# actions)
        layer_n = (num_input, *num_hidden, num_output)
        # Question: might this not work? Try without ReLU's like in the RL course
        for i in range(len(layer_n[:-1])):
            layers.append(nn.Linear(layer_n[i], layer_n[i + 1]))
            if i < len(layer_n) - 2:
                # Default ReLU
                layers.append(nn.ReLU())
        # Last layer is log_softmax
        layers.append(nn.Linear(layer_n[i + 1], 16))
        layers.append(nn.LogSoftmax())
        self.network = nn.Sequential(*layers)

        # Question: what is this line for?
        self.d_type = self.network.state_dict()["0.weight"].type()

    def forward(self, x):
        return self.network(x)


class Simple_PG:
    def __init__(
        self,
        input_size,
        output_size,
        loss_function,
        num_hidden=(64,),
        optimizer=SGD,
        lr=0.001,
        gamma=0.99,
        epsilon_delta=0.0001,
        epsilon_min=0.05,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_delta = epsilon_delta
        self.epsilon = 1.0
        self.policy_network = PolicyNetwork(
            input_size, output_size, num_hidden=num_hidden
        )
        self.optimizer = optimizer(self.policy_network.parameters(), lr=lr)
        self.loss_function = loss_function
        self.clip_norm_value = 2.0

    def check_obs(self, obs):
        """
        Before calling `predict()` we check if the given obs is a tensor.
        """
        # Check that obs is a tensor
        if type(obs) == torch.tensor:
            pass
        elif type(obs) == np.ndarray:
            obs = torch.from_numpy(obs)
        else:
            raise ValueError
        # Check tensor is 2D
        assert len(obs.shape) < 3
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, obs.shape[-1])
        # Check feature dimension is correct
        assert obs.shape[-1] == self.input_size
        # Check data type is correct
        if obs.type() != self.policy_network.d_type:
            obs = obs.type(self.policy_network.d_type)
        return obs

    def predict(self, obs, eval=False):
        obs = self.check_obs(obs)
        if eval:
            with torch.no_grad():
                # Returns action, and log probs for each action
                log_probs = self.policy_network(obs)
                # 'Fully greedy'
                return torch.argmax(log_probs).item()

        log_probs = self.policy_network(obs)
        action = torch.multinomial(log_probs.exp(), 1).item()

        return action, log_probs

    def train(self, obses_t, actions, log_prob_actions, rewards, obses_tp1, dones):
        # Convert to torch.tensor
        log_prob_actions = torch.stack(log_prob_actions, dim=0)
        rewards = torch.from_numpy(rewards).to(torch.float)

        # Calculate Loss
        episode_return = 0.0
        returns = []

        for r in np.array(rewards)[::-1]:
            episode_return = r + self.gamma * episode_return
            returns.append(episode_return)
        returns = torch.tensor(returns[::-1])
        returns -= torch.mean(returns)
        returns /= torch.std(returns)
        loss = -(returns * log_prob_actions).sum()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.clip_norm_value
        )
        self.optimizer.step()

        return loss.detach().item()
