import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
np.random.seed(1)
torch.manual_seed(1)

class Actor_Gaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor_Gaussian, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(
            torch.zeros(1, action_dim))  # We use 'nn.Paremeter' to train log_std automatically

    def forward(self, s):
        mean = self.actor_net(s)
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Actor_Gaussian_HRL(nn.Module):  # Q网络
    def __init__(self, args):
        super(Actor_Gaussian_HRL, self).__init__()
        self.args = args
        self.S_actor_net = Actor_Gaussian(self.args.critic_sli, args.S_action_n, args.hidden_width)
        self.U_actor_net = Actor_Gaussian(args.S_action_n + args.state_n, args.U_action_n, args.hidden_width)

    def forward(self, state):
        S_act = self.S_actor_net(state[:,:self.args.critic_sli])
        U_act = self.U_actor_net(torch.cat([S_act, state], -1))
        return torch.cat([S_act, U_act], -1)
    
    def get_dist(self, s):
        S_mean = self.S_actor_net(s[:,:self.args.critic_sli])
        S_log_std = self.S_actor_net.log_std.expand_as(S_mean)  # To make 'log_std' have the same dimension as 'mean'
        S_std = torch.exp(S_log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        
        U_mean = self.U_actor_net(torch.cat([S_mean, s], -1))
        U_log_std = self.U_actor_net.log_std.expand_as(U_mean)  # To make 'log_std' have the same dimension as 'mean'
        U_std = torch.exp(U_log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        
        dist = Normal(torch.cat([S_mean, U_mean], -1), torch.cat([S_std, U_std], -1))  # Get the Gaussian distribution
        return dist
    
    def get_dist_S(self, s):
        return self.S_actor_net.get_dist(s[:,:self.args.critic_sli])
    
    def get_dist_U(self, S_act, s):
        return self.U_actor_net.get_dist(torch.cat([S_act, s], -1))

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc11 = nn.Linear(args.critic_sli, int(args.hidden_width / 2))
        self.fc12 = nn.Linear(int(args.hidden_width / 2), int(args.hidden_width / 2))
        self.fc13 = nn.Linear(int(args.hidden_width / 2), int(args.hidden_width / 2))

        self.fc21 = nn.Linear(args.state_n-args.critic_sli, int(args.hidden_width / 2))
        self.fc22 = nn.Linear(int(args.hidden_width / 2), int(args.hidden_width / 2))
        self.fc23 = nn.Linear( int(args.hidden_width / 2),  int(args.hidden_width / 2))

        self.fc4 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh


    def forward(self, s, numpy_type = False):
        if numpy_type:
            s = torch.from_numpy(s).float()
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        s1 = s[:, :self.args.critic_sli]
        s1 = self.activate_func(self.fc11(s1))
        s1 = self.activate_func(self.fc12(s1))
        s1 = self.activate_func(self.fc13(s1))

        s2 = s[:, self.args.critic_sli:]
        s2 = self.activate_func(self.fc21(s2))
        s2 = self.activate_func(self.fc22(s2))
        s2 = self.activate_func(self.fc23(s2))

        v_s = self.fc4(torch.cat([s1, s2], -1))
        return v_s

