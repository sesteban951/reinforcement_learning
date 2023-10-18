import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """
    Abstract base class for a policy network that takes continuous observations
    and outputs continuous actions.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Check that observation and action spaces are continuous
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Box)

        self.input_size = observation_space.shape[0]
        self.output_size = action_space.shape[0]

    def forward(self, x):
        """
        Forward pass through the policy network.

        Args:
            x: The observation
        
        Returns:
            The mean and standard deviation of the action distribution
        """
        raise NotImplementedError

    def sample(self, x):
        """
        Given an observation, sample an action from the action distribution.

        Args:
            x: The observation

        Returns:
            The action and the log probability of the action
        """
        raise NotImplementedError
    
    def reset(self):
        """
        Reset the hidden state, e.g. between episodes, if necessary.
        """
        pass

class MlpPolicy(PolicyNetwork):
    """
    A simple policy network based on a Multilayer Perceptron (MLP).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Define the mean network
        self.mean_network = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

    def forward(self, x):
        mean = self.mean_network(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, x):
        mean, std = self.forward(x)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        return action, log_prob
    
class RnnPolicy(PolicyNetwork):
    """
    A simple policy network based on a Recurrent Neural Network (RNN).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Define the mean network as an RNN
        self.state_size = 64  # Size of the hidden state
        self.recurrent_network = nn.RNN(self.input_size, self.state_size, nonlinearity='tanh', batch_first=True)
        self.output_network = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Linear(self.state_size, self.output_size))

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

        # Reset the hidden state
        self.reset()

    def forward(self, x):
        _, hidden = self.recurrent_network(x.unsqueeze(0), self.hidden_state)
        self.hidden_state = hidden.detach()
        mean = self.output_network(hidden.squeeze(0))
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, x):
        mean, std = self.forward(x)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        return action, log_prob
    
    def reset(self):
        self.hidden_state = torch.zeros(1, 64)

class KoopmanPolicy(PolicyNetwork):
    """
    A recurrent policy network based on Koopman theory. The controller is
    treated as a linear system,

        x_{t+1} = Ax_t + Bu_t,
        y_t = Cx_t + Du_t,

    where u_t is the input (observations), y_t is the output (actions), and x_t
    is the state. Koopman tells us that any nonlinear system can be represented 
    as a linear system in an infinite-dimensional space, and the perfect
    controller can be described as a nonlinear system, so we'll learn a
    finite-dimensional linear approximation.
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Decide on the size of the hidden state x
        self.hidden_state_size = 64

        # Linear system matrices
        self.A = nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.B = nn.Linear(self.input_size, self.hidden_state_size, bias=False)
        self.C = nn.Linear(self.hidden_state_size, self.output_size, bias=False)
        self.D = nn.Linear(self.input_size, self.output_size, bias=False)

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

        # Allocate the hidden state
        self.reset()

    def forward(self, u):
        # Compute the output (mean) based on the current state
        y = self.C(self.x) + self.D(u)

        # Advance the linear system dynamics
        self.x = self.A(self.x) + self.B(u)

        # Return the mean and standard deviation of the action distribution
        std = torch.exp(self.log_std)
        return y, std
    
    def sample(self, x):
        mean, std = self.forward(x)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        return action, log_prob
    
    def reset(self):
        self.x = torch.zeros(self.hidden_state_size)