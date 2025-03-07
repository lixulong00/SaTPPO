import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
np.random.seed(1)
torch.manual_seed(1)

class Actor_Gaussian(nn.Module):
    """
    Gaussian Actor network for continuous action spaces.
    
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_width (int): Width of the hidden layers.
    """
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor_Gaussian, self).__init__()
        # Define the network structure for the actor
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, action_dim),
            nn.Tanh()
        )
        # Define log_std as a learnable parameter to ensure std > 0
        self.log_std = nn.Parameter(
            torch.zeros(1, action_dim))  

    def forward(self, state):
        """
        Forward pass to compute the mean of the action distribution.
        
        Args:
            state (Tensor): Current state tensor.
        
        Returns:
            mean (Tensor): Mean of the action distribution.
        """
        mean = self.actor_net(state)
        return mean

    def get_dist(self, state):
        """
        Get the Gaussian distribution for the given state.
        
        Args:
            state (Tensor): Current state tensor.
        
        Returns:
            dist (Distribution): Gaussian distribution of actions.
        """
        mean = self.forward(state)
        log_std = self.log_std.expand_as(mean)  
        std = torch.exp(log_std)  
        dist = Normal(mean, std)  
        return dist


class Actor_Gaussian_TS(nn.Module):  
    """
    Gaussian Actor network with two stages for complex action spaces.
    
    Args:
        args (namespace): Contains state_dim, action_S_dim, action_U_dim, and hidden_width.
    """
    def __init__(self, args):
        super(Actor_Gaussian_TS, self).__init__()
        # Define two separate actor networks for different action components
        self.S_actor_net = Actor_Gaussian(args.state_dim, args.action_S_dim, args.hidden_width)
        self.U_actor_net = Actor_Gaussian(args.state_dim + args.action_S_dim, args.action_U_dim, args.hidden_width)

    def forward(self, state):
        """
        Forward pass to compute the concatenated action.
        
        Args:
            state (Tensor): Current state tensor.
        
        Returns:
            Tensor: Concatenated action tensor.
        """
        S_act = self.S_actor_net(state)
        U_act = self.U_actor_net(torch.cat([S_act, state], -1))
        return torch.cat([S_act, U_act], -1)
    
    def get_dist(self, state):
        """
        Get the joint Gaussian distribution for the two-stage action.
        
        Args:
            state (Tensor): Current state tensor.
        
        Returns:
            dist (Distribution): Joint Gaussian distribution of actions.
        """
        S_mean = self.S_actor_net(state)
        S_log_std = self.S_actor_net.log_std.expand_as(S_mean)  
        S_std = torch.exp(S_log_std)  
        
        U_mean = self.U_actor_net(torch.cat([S_mean, state], -1))
        U_log_std = self.U_actor_net.log_std.expand_as(U_mean)  
        U_std = torch.exp(U_log_std)  
        
        dist = Normal(torch.cat([S_mean, U_mean], -1), torch.cat([S_std, U_std], -1))  
        return dist
    
    def get_dist_S(self, state):
        """
        Get the Gaussian distribution for the first stage action.
        
        Args:
            state (Tensor): Current state tensor.
        
        Returns:
            Distribution: Gaussian distribution of the first stage action.
        """
        return self.S_actor_net.get_dist(state)
    
    def get_dist_U(self, S_act, state):
        """
        Get the Gaussian distribution for the second stage action.
        
        Args:
            S_act (Tensor): Action from the first stage.
            state (Tensor): Current state tensor.
        
        Returns:
            Distribution: Gaussian distribution of the second stage action.
        """
        return self.U_actor_net.get_dist(torch.cat([S_act, state], -1))



class Critic(nn.Module):
    """
    Critic network to evaluate the value of a state-action pair.
    
    Args:
        args (namespace): Contains state_dim, action_dim, and hidden_width.
    """
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        # Define the network structure for the critic
        self.critic_net = nn.Sequential(
            nn.Linear(args.state_dim + args.action_dim, args.hidden_width),
            nn.ReLU(),
            nn.Linear(args.hidden_width, args.hidden_width),
            nn.ReLU(),
            nn.Linear(args.hidden_width, args.hidden_width),
            nn.ReLU(),
            nn.Linear(args.hidden_width, 1),
            nn.Tanh()
        )

    def forward(self, state, numpy_type = False):
        """
        Forward pass to compute the value of the given state(s).
        
        Args:
            state (Tensor or ndarray): Current state tensor or numpy array.
            numpy_type (bool): Flag indicating if the input is a numpy array.
        
        Returns:
            v_s (Tensor): Value of the state(s).
        """
        if numpy_type:
            state = torch.from_numpy(state).float()
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        v_s = self.critic_net(state)
        return v_s