import torch.nn as nn
import torch


def init_weights(m, bias=0.01):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(bias)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.LeakyReLU(),   # nn.Tanh(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),   # nn.Tanh(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),   # nn.Tanh(),
            nn.Linear(32, self.output_dim))

    def forward(self, state):
        """
        Args:
            state (torch.tensor): of shape (batch_size, number of features
        """
        qvals = self.fc(state)
        return qvals


class AgentDQN:
    """
    Agent class encapsulates the algorithm to make decisions (policy).
    Here DQN and Double-DQN agent lives.
    """

    def __init__(
        self,
        env,
        buffer,
        loss_function,
        optimizer,
        lr,
        gamma,
        epsilon_delta,
        epsilon_min,
        batch_size,
        epsilon=1,
        tau=0.5,
        double_dqn=False,
        target_update=True,
        sequence_length=None,
        seed=None,
    ):
        torch.manual_seed(seed)
        self.env = env
        self.replay_buffer = buffer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.double_dqn = double_dqn
        self.learning_rate = lr
        # Responsible for how much the control network get updates at each call of `update`
        self.tau = tau
        # Whether to update the target on this iteration
        self.target_update = target_update
        self.loss_function = loss_function
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.epsilon_min = epsilon_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_norm_value = 1.0

        self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        # Apply some other parameter initiatlization
        # init_weights(self.model)
        # init_weights(self.target_model)

        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()

        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

    def get_epsilon(self, it, periods_zero_epsilon=60):
        """
        TODO: how to select this epsilon decay parameters?
        """
        eps = 1 - (it * 1/periods_zero_epsilon)

        return eps if eps > 0 else 0.05

    def select_action(self, state, epsilon, dt_state=None, batch_size=1, sequence_length=1):
        # No need to update gradients for this
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            q = self.model.forward(state)
            r = torch.rand(1).item()
            if r < epsilon:
                return int(len(q) * r / epsilon)
            else:
                # select indicies
                return torch.max(q, 1)[1].item()

    def compute_q_val(self, model, state, action):
        # This is what we will be updating
        return model(torch.FloatTensor(state))[range(len(state)), action]

    def compute_target(self, model, reward, next_state, done, discount_factor):
        targets = reward + discount_factor * model(next_state).max(1)[0] * (1 - done.float())
        # If done, Terminal state q val is zero for every action, target is just the reward then
        return targets

    def train(self, target_update=False, **kwargs):
        # Don't learn without some decent experience
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Random transition batch is taken from experience replay memory
        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, reward, next_state, done = transitions

        # Convert to PyTorch and define types
        state = torch.tensor(state, dtype=torch.float)
        # Need 64 bit to use them as index
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # Boolean
        done = torch.tensor(done, dtype=torch.uint8)

        # Compute the q value
        q_val = self.compute_q_val(self.model, state, action)

        if self.double_dqn:
            # Get the Q values for best actions in obs_tp1
            # Based on the current Q network: max(Q(s', a', theta_i)) wrt a'
            q_tp1_values = self.model(next_state).detach()
            _, a_prime = q_tp1_values.max(1)
            # Get Q values from frozen network for next state and chosen action
            # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
            q_target_tp1_values = self.target_model(next_state).detach()
            q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
            q_target_s_a_prime = q_target_s_a_prime.squeeze()
            # if current state is end of episode, then there is no next Q value
            q_target_s_a_prime = (1 - done) * q_target_s_a_prime
            target = reward + self.gamma * q_target_s_a_prime
        else:
            # Don't compute gradient info for the target (semi-gradient)
            with torch.no_grad():
                target = self.compute_target(self.target_model, reward, next_state, done, self.gamma)

        # Loss is measured from error between current and newly expected Q values
        loss = self.loss_function(input=q_val, target=target)

        # Backpropagation of loss. You still have old gradients saved in pytorch
        # backward() function accumulates gradients, and you don’t want to mix up gradients between minibatches
        # So you zero_grad()
        self.optimizer.zero_grad()

        # Backprop
        loss.backward()

        ##################
        # Clip gradients
        ##################
        # torch.nn.utils.clip_grad_norm_(
        #     self.model.parameters(), self.clip_norm_value
        # )
        # torch.nn.utils.clip_grad_norm_(
        #     self.target_model.parameters(), self.clip_norm_value
        # )

        # Perform a parameter update based on the current gradient
        self.optimizer.step()

        if self.target_update:
            self.target_model.load_state_dict(self.model.state_dict())
            if self.double_dqn:
                # Update target network with learning rate tau
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Returns a Python scalar, and releases history
        return loss.item()
