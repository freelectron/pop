import torch.nn as nn
import torch

torch.manual_seed(25)


def init_weights(m, bias=0.01):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(bias)


class RDQN(nn.Module):
    """
    Many-to-one.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(RDQN, self).__init__()
        # Number of features
        self.input_dim = input_dim[0]
        # Hidden dimensions for LSTM cell
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # The LSTM takes inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, seq, batch_size=1, sequence_length=1):
        """
        I do many-to-one LSTM.

        Args:
            seq (torch.tensor): of shape [seq_length, n_features]

        torch.nn.LSTM returns: output, (h_n, c_n)
             output: tensor containing the output features (h_t) from the last layer of the LSTM, for each time step in
                     the sequence.
             h_n: shape (num_layers, batch, hidden_size), tensor containing the hidden state for t = seq_len
             c_n: shape (num_layers, batch, hidden_size), tensor containing the cell state for t = seq_len.
        """
        # lstm_out is of shape (seq_len, batch, hidden_size)
        lstm_output, (lstm_hidden, _) = self.lstm(seq.view(batch_size, sequence_length, self.input_dim))
        qvals = self.fc(lstm_hidden.transpose(0, 1).view(batch_size, 1, self.hidden_dim)[:, -1, :])  #.squeeze())  #

        return qvals


class AgentRDQN:
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
        epsilon=1,
        tau=1.0,
        double_dqn=False,
        target_update=True,
        batch_size=None,
        sequence_length=None,
        seed=None,
    ):
        torch.manual_seed(seed)
        self.env = env
        self.replay_buffer = buffer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
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

        self.model = RDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model = RDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        # Apply some other parameter initiatlization
        # init_weights(self.model)
        # init_weights(self.target_model)

        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()

        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

    def get_epsilon(self, it, periods_zero_epsilon=100):
        """
        TODO: how to select this epsilon decay parameters?
        """
        eps = 1 - (it * 1/periods_zero_epsilon)

        return eps if eps > 0 else 0.05

    def select_action(self, state, epsilon, batch_size=1, sequence_length=1, dt_state=None):
        # No need to update gradients for this
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            q = self.model.forward(state, batch_size=batch_size, sequence_length=sequence_length)
            r = torch.rand(1).item()
            if r < epsilon:
                # q is always of shape [batch_size, n_actions]
                return int(q.shape[1] * r / epsilon)
            else:
                # select indicies
                return torch.max(q, 1)[1].item()

    def compute_q_val(self, model, state, action):
        # This is what we will be updating
        return model(torch.FloatTensor(state), batch_size=self.batch_size, sequence_length=self.sequence_length)[range(self.batch_size), action]

    def compute_target(self, model, reward, next_state, done, discount_factor):
        targets = reward + discount_factor * model(next_state, batch_size=self.batch_size,
                                                   sequence_length=self.sequence_length).max(1)[0] * (1 - done.float())
        # If done, Terminal state q val is zero for every action, target is just the reward then
        return targets

    def train(self, target_update=False, **kwargs):
        # Don't learn without some decent experience
        if len(self.replay_buffer) < self.batch_size * self.sequence_length:
            return None

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = list(), list(), list(), list(), list()
        for _ in range(self.batch_size):
            # Random transition batch is taken from experience replay memory
            transitions = self.replay_buffer.sample(self.sequence_length, sequential=True)
            state, action, reward, next_state, done = transitions
            # Convert to PyTorch and define types
            state_batch.append(torch.tensor(state, dtype=torch.float))
            # Need 64 bit to use them as index
            action_batch.append(torch.tensor(action, dtype=torch.int64))
            next_state_batch.append(torch.tensor(next_state, dtype=torch.float))
            reward_batch.append(torch.tensor(reward, dtype=torch.float))
            done_batch.append(torch.tensor(done, dtype=torch.uint8))
        state = torch.stack(state_batch).squeeze()
        action = torch.stack(action_batch)[:, -1]
        reward = torch.stack(reward_batch)[:, -1]
        next_state = torch.stack(next_state_batch).squeeze()
        done = torch.stack(done_batch)[:, -1]

        # Compute the q value
        q_val = self.compute_q_val(self.model, state, action)

        if self.double_dqn:
            # Get the Q values for best actions in obs_tp1
            # Based on the current Q network: max(Q(s', a', theta_i)) wrt a'
            q_tp1_values = self.model(next_state, batch_size=self.batch_size,
                                      sequence_length=self.sequence_length).detach()
            _, a_prime = q_tp1_values.max(1)
            # Get Q values from frozen network for next state and chosen action
            # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
            q_target_tp1_values = self.target_model(next_state, batch_size=self.batch_size,
                                                    sequence_length=self.sequence_length).detach()
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
        # All the loss functions in pytorch assume that the first dimension is the batch size
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
            # Udpate without learning rate
            self.target_model.load_state_dict(self.model.state_dict())
            if self.double_dqn:
                # Update target network with learning rate tau
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Returns a Python scalar, and releases history
        return loss.item()