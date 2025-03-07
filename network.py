import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
np.random.seed(1)
torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_lr = 1e-5
qf_lr = 1e-4
hidden_sizes = 128
gamma = 0.95
batch_size = 256
ReplayBuffer_size = 100000
expl_before = 2000
tau = 0.01
exp_noise = 0.1
policy_noise = 0.2
tot_exp = 10000


LOG_STD_MAX = 2
LOG_STD_MIN = -20
class ReparamGaussianPolicy(nn.Module):
 def __init__(self,
              input_size,
              output_size,
              hidden_sizes=256,
              ):
  super(ReparamGaussianPolicy, self).__init__()
  self.linear1 = nn.Linear(input_size, hidden_sizes)
  self.linear2 = nn.Linear(hidden_sizes, hidden_sizes)
  # Set output layers
  self.mu_layer = nn.Linear(hidden_sizes, output_size)
  self.log_std_layer = nn.Linear(hidden_sizes, output_size)

 def clip_but_pass_gradient(self, x, l=-1., u=1.):
  clip_up = (x > u).float()
  clip_low = (x < l).float()
  clip_value = (u - x) * clip_up + (l - x) * clip_low
  return x + clip_value.detach()

 def apply_squashing_func(self, mu, pi, log_pi):
  mu = torch.tanh(mu)
  pi = torch.tanh(pi)
  # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
  log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6), dim=-1)
  return mu, pi, log_pi

 def forward(self, state):
  x = F.relu(self.linear1(state))
  x = F.relu(self.linear2(x))
  mu = self.mu_layer(x)
  log_std = torch.tanh(self.log_std_layer(x))
  log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
  std = torch.exp(log_std)

  # https://pytorch.org/docs/stable/distributions.html#normal
  dist = Normal(mu, std)
  pi = dist.rsample()  # Reparameterization trick (mean + std * N(0,1))
  log_pi = dist.log_prob(pi).sum(dim=-1)
  mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)

  # Make sure outputs are in correct range
#   mu = mu * self.output_limit
#   pi = pi * self.output_limit

  return mu, pi, log_pi
 def get_act(self, state, numpy_type =False):
    if numpy_type:
        state = torch.from_numpy(state).float().to(device)
    act = self.forward(state)[1]
    if numpy_type:
        act = act.detach().cpu().numpy()
    return act

 def get_act_val(self, state, numpy_type =False):
     if numpy_type:
        state = torch.from_numpy(state).float().to(device)
     act = self.forward(state)[0]
     if numpy_type:
        act = act.detach().cpu().numpy()
     return act


class Critic_Net(nn.Module):  # Q网络
 def __init__(self, input_dim, hidden_size, init_w=3e-3):
  super(Critic_Net, self).__init__()
  self.linear1 = nn.Linear(input_dim, hidden_size)
  self.linear2 = nn.Linear(hidden_size, hidden_size)
  self.linear2_ = nn.Linear(hidden_size, hidden_size)
  self.linear3 = nn.Linear(hidden_size, 1)

 def forward(self, state, action, numpy_type=False):
  if numpy_type:
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
  x = torch.cat([state, action], -1)  # 将state、action拼接在一起
  x = F.relu(self.linear1(x))
  x = F.relu(self.linear2(x))
  x = F.relu(self.linear2_(x))
  x = self.linear3(x)
  return x


class Critic_Net1(nn.Module):  # Q网络
 def __init__(self, input_dim, hidden_size, init_w=3e-3):
  super(Critic_Net, self).__init__()
  self.linear1 = nn.Linear(input_dim, hidden_size)
  self.linear2 = nn.Linear(hidden_size, hidden_size)
  self.linear2_ = nn.Linear(hidden_size, hidden_size)
  self.linear3 = nn.Linear(hidden_size, 1)

 def forward(self, state, action, numpy_type=False):
  if numpy_type:
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
  x = torch.cat([state, action], -1)  # 将state、action拼接在一起
  x = F.relu(self.linear1(x))
  x = F.relu(self.linear2(x))
  x = F.relu(self.linear2_(x))
  x = self.linear3(x)
  return x



class Actor_Net(nn.Module):  # Q网络
    def __init__(self, input_dim,output_dim, hidden_size, noise=exp_noise, init_w = 1e-3):
        super(Actor_Net, self).__init__()
        self.noise = noise
        self.linear1 = nn.Linear(input_dim, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_act(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)
        act = (act + torch.randn_like(act) * self.noise ).clamp(-1, 1)
        if numpy_type:
            act = act.detach().cpu().numpy()
        return act

    def get_act_val(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)
        if numpy_type:
            act = act.detach().cpu().numpy()   
        return self.forward(state)


class DQN_net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(DQN_net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
    def get_act(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)
        if numpy_type:
            act = act.detach().cpu().numpy()
        return act
    
    def get_act_val(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)
        if numpy_type:
            act = act.detach().cpu().numpy()
        return self.forward(state)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.buffer.append([state, action, reward, next_state, [1 if done else 0]])

    def sample(self, batch_size):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
            *random.sample(self.buffer, batch_size))
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def __len__(self):
        return len(self.buffer)


    def __len__(self):
        return len(self.buffer)


class ReparamGaussianPolicy_HRL(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                hidden_size=256,
                ):
        super(ReparamGaussianPolicy_HRL, self).__init__()
        self.S_actor_net = ReparamGaussianPolicy(input_dim, output_dim[0], hidden_size)
        self.U_actor_net = ReparamGaussianPolicy(input_dim+output_dim[0], output_dim[1], hidden_size)

    def forward(self, state):
        S_mu, S_pi, S_log_pi = self.S_actor_net(state)
        U_mu, U_pi, U_log_pi = self.U_actor_net(torch.cat([S_pi, state], -1))
        mu = torch.cat([S_mu, U_mu], -1)
        pi = torch.cat([S_pi, U_pi], -1)
        log_pi = S_log_pi + U_log_pi
        return mu, pi, log_pi
    
    def get_act(self, state, numpy_type =False):
        if numpy_type:
            state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)[1]
        if numpy_type:
            act = act.detach().cpu().numpy()
        return act
    def get_act_U(self, S_act, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
           S_act = torch.from_numpy(S_act).float().to(device)
        U_act = self.U_actor_net(torch.cat([S_act, state], -1))[1]
        if numpy_type:
            U_act = U_act.detach().cpu().numpy()
        return U_act

    def get_act_S(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        S_act = self.S_actor_net(state)[1]
        if numpy_type:
            S_act = S_act.detach().cpu().numpy()
        return S_act


    def get_act_val(self, state, numpy_type =False):
        if numpy_type:
            state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)[0]
        if numpy_type:
            act = act.detach().cpu().numpy()
        return act
    

    def get_act_val_U(self, S_act, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
           S_act = torch.from_numpy(S_act).float().to(device)
        U_act = self.U_actor_net(torch.cat([S_act, state], -1))[0]
        if numpy_type:
            U_act = U_act.detach().cpu().numpy()
        return U_act

    def get_act_val_S(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        S_act = self.S_actor_net(state)[0]
        if numpy_type:
            S_act = S_act.detach().cpu().numpy()
        return S_act

class Actor_Net_HRL(nn.Module):  # Q网络
    def __init__(self, input_dim, output_dim, hidden_size, noise=exp_noise):
        super(Actor_Net_HRL, self).__init__()
        self.noise = exp_noise
        self.S_actor_net = Actor_Net(input_dim, output_dim[0], hidden_size, noise)
        self.U_actor_net = Actor_Net(input_dim+output_dim[0], output_dim[1], hidden_size, noise)

    def forward(self, state):
        S_act = self.S_actor_net(state)
        U_act = self.U_actor_net(torch.cat([S_act, state], -1))
        return torch.cat([S_act, U_act], -1)

    def get_act(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)
        act = (act + torch.randn_like(act) * self.noise).clamp(-1, 1)
        if numpy_type:
            act = act.detach().cpu().numpy()
        return act
    
    def get_act_U(self, S_act, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
           S_act = torch.from_numpy(S_act).float().to(device)
        U_act = self.U_actor_net(torch.cat([S_act, state], -1))
        U_act = (U_act + torch.randn_like(U_act) * self.noise ).clamp(-1, 1)
        if numpy_type:
            U_act = U_act.detach().cpu().numpy()
        return U_act

    def get_act_S(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        S_act = self.S_actor_net(state)
        S_act = (S_act + torch.randn_like(S_act) * self.noise ).clamp(-1, 1)
        if numpy_type:
            S_act = S_act.detach().cpu().numpy()
        return S_act
    
    def get_act_val(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        act = self.forward(state)
        if numpy_type:
            act = act.detach().cpu().numpy()   
        return self.forward(state)
    
    def get_act_val_U(self, S_act, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
           S_act = torch.from_numpy(S_act).float().to(device)
        U_act = self.U_actor_net(torch.cat([S_act, state], -1))
        if numpy_type:
            U_act = U_act.detach().cpu().numpy()
        return U_act

    def get_act_val_S(self, state, numpy_type =False):
        if numpy_type:
           state = torch.from_numpy(state).float().to(device)
        S_act = self.S_actor_net(state)
        if numpy_type:
            S_act = S_act.detach().cpu().numpy()
        return S_act


class CRF_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CRF_GRU, self).__init__()
        # self.gru = nn.GRUCell(input_dim, hidden_dim, False)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, numpy_type =False):
        if numpy_type:
            state = torch.from_numpy(state).float().to(device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def te_st(my_env, agentes):
    state_ = my_env.reset()
    episode_reward = 0
    done = False
    while not done:
        act = agentes.actors_net.get_act_val(state_, numpy_type=True)
        next_state_, r, done, info = my_env.step(act)
        state_ = next_state_
        episode_reward += r
        if done:
            break
    return episode_reward


def save_info(agentes, totle_info, alg):
    torch.save(agentes.actors_net.state_dict(), "./net/actor_" + alg + ".pth")
    for key in totle_info.keys():
        np.save("./date/" + str(key) + "_"+alg + ".npy", totle_info[key])


def main(my_env, agentes, alg):
    totle_info = {
        "reward": [],
    }
    state = my_env.reset()
    while True:
        if len(agentes.replayBuffer) < agentes.expl_before:
            act = np.random.uniform(-1, 1, (my_env.action_n,))
            next_state, r, done, info = my_env.step(act)
            agentes.replayBuffer.push(state, act, r, next_state, done)
            state = next_state
            if done:
                state = my_env.reset()
        else:
            break
    
    for exp in range(tot_exp):
        state = my_env.reset()
        done = False
        while not done:
            act = agentes.actors_net.get_act(state, numpy_type=True)
            next_state, r, done, info = my_env.step(act)
            agentes.replayBuffer.push(state, act, r, next_state, done)
            state = next_state
            if len(agentes.replayBuffer) > agentes.batch_size:
                agentes.update_policy()
            if done:
                break

        episode_reward = te_st(my_env, agentes)
        totle_info["reward"].append(episode_reward)
        print('alg:', alg,'episode:', exp, '  episode_reward:', episode_reward)
        if exp % 10 == 0:
            save_info(agentes, totle_info, alg)


