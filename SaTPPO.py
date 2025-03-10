import sys
import torch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from ppo_network import *
sys.path.append('..')
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the replay buffer class for storing experience data
class ReplayBuffer:
    """
    Initialize the replay buffer
    :param args: Contains the parameters required for the replay buffer, such as batch size, state dimension, action dimension, etc.
    """
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    """
    Store experience data into the replay buffer
    :param s: Current state
    :param a: Action taken
    :param a_logprob: Log probability of the action
    :param r: Reward received
    :param s_: Next state
    :param dw: Flag for death or win (no next state)
    :param done: Flag for episode termination
    """
    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    """
    Convert numpy arrays to PyTorch tensors
    :return: Tuple of tensors containing states, actions, action log probabilities, rewards, next states, death/win flags, and done flags
    """
    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        return s, a, a_logprob, r, s_, dw, done


# Define the SaTPPO class
class SaTPPO():
    """
    Initialize the SaTPPO algorithm
    :param args: Contains the parameters required for the algorithm, such as batch size, learning rate, etc.
    """
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.actor = Actor_Gaussian_TS(args.state_n, [args.S_action_n, args.U_action_n], hidden_size=args.hidden_width)
        self.critic = Critic(args)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
    
    """
    Evaluate the policy and return the mean action for a given state
    :param s: State
    :return: Mean action
    """
    def evaluate_S(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor.S_actor_net(s).detach()
        return a.numpy()

    """
    Evaluate the policy and return the mean action for a given state and sub-action
    :param S_act: Sub-action
    :param s: State
    :return: Mean action
    """
    def evaluate_U(self, S_act, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor.U_actor_net(torch.cat([S_act, s], -1)).detach()
        return a.numpy()
    
    """
    Choose an action based on the current policy for a given state
    :param s: State
    :return: Action and its log probability
    """
    def choose_action_S(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist_S(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -1, 1)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy(), a_logprob.numpy()

    """
    Choose an action based on the current policy for a given state and sub-action
    :param S_act: Sub-action
    :param s: State
    :return: Action and its log probability
    """
    def choose_action_U(self, S_act, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist_U(S_act, s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -1, 1)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy(), a_logprob.numpy()

    """
    Save the model parameters
    """
    def save_model(self):
        net = {'PPO_actor_net':self.actor, 'PPO_critic_net':self.critic}
        utils.save_model(net, self.args.full_save_path + '/model_PPO' , self.args.save_network_type)

    """
    Update the actor and critic networks using experience from the replay buffer
    :param replay_buffer: Replay buffer containing the experience data
    """
    def update(self, replay_buffer):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize network for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index, :])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index, :])
                # In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index, :].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

# Test the performance of the agent
def te_st(my_env, agent):
    """
    Test the performance of the agent in the environment.

    Parameters:
    - my_env: The environment object where the agent acts.
    - agent: The agent object to be tested.

    Returns:
    - episode_reward: The cumulative reward obtained during the episode.
    - episode_info: A dictionary containing additional information from each step of the episode.
    """
    episode_reward = 0
    episode_info = {}
    s = my_env.reset()
    done = False
    S_action = None
    while not done:
        new_S_action = agent.evaluate_S(s)
        if S_action is None:
            S_action = new_S_action
        tmp_new_state, cost = my_env.new_state_cost(new_S_action)
        if agent.critic(tmp_new_state, numpy_type=True).detach().numpy()[0] + cost \
                > agent.critic(s, numpy_type=True).detach().numpy()[0]:
            S_action = new_S_action
        U_action = agent.evaluate_U(torch.tensor(S_action, dtype=torch.float), s)
        act = np.append(S_action, U_action, -1)
        s_, r, done, info = my_env.step(act[0])
        episode_reward += r
        if done:
            break
        s = s_
        for key in info.keys():
            if key not in episode_info.keys():
                episode_info[key] = []
            episode_info[key].append(info[key])
    return episode_reward, episode_info

# Main function for training the agent
def main(args, my_env):
    """
    The main function for training the agent.

    Parameters:
    - args: Command line arguments and configuration parameters.
    """
    
    # Initialize the experience replay buffer and the PPO agent
    replay_buffer = ReplayBuffer(args)
    agent = SaTPPO(args)

    # Initialize a dictionary to store training information
    totle_info = {
        "reward": [],
    }

    # Start training for a fixed number of episodes
    for exp in range(args.episode_number_max):
        episode_reward = 0
        s = my_env.reset()
        done = False
        S_action = None
        S_a_logprob = None
        while not done:
            # obtain long-timescale actions
            new_S_action, new_S_a_logprob = agent.choose_action_S(s)
            if S_action is None:
                S_action = new_S_action
                S_a_logprob = new_S_a_logprob
            # Build new system state based on long-timescale actions and calculate the reconfiguration cost 
            tmp_new_state, cost = my_env.new_state_cost(new_S_action)

            # Decide whether to adopt the new long-timescale action based on the critic's evaluation
            if agent.critic(tmp_new_state, numpy_type=True).detach().numpy()[0] - \
                agent.critic(s, numpy_type=True).detach().numpy()[0] > cost:
                S_action = new_S_action
                S_a_logprob = new_S_a_logprob
            
            # Obtain short-timescale actions 
            U_action, U_a_logprob = agent.choose_action_U(torch.tensor(S_action, dtype=torch.float), s)
            
            # Obtain joint action 
            act = np.append(S_action, U_action, -1)
            a_logprob = np.append(S_a_logprob, U_a_logprob, -1)

            # Execute the action and observe the next state, reward, and other information
            s_, r, done, _ = my_env.step(act[0])
            episode_reward += r
            if done:
                dw = True
            else:
                dw = False
            # Store the experience
            replay_buffer.store(s, act, a_logprob, r, s_, dw, done)
            s = s_
            # Update the agent when the replay buffer is full
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer)
                replay_buffer.count = 0
        # Test and save training information at fixed intervals
        if exp % args.test_per_episode == 0:
            episode_reward, episode_info = te_st(my_env, agent)
            totle_info["reward"].append(episode_reward)
            for key in episode_info.keys():
                if key not in totle_info.keys():
                    totle_info[key] = []
                totle_info[key].append(episode_info[key])
  
            tmp_str = f'alg:{args.alg}, episode:{exp}, episode_reward:{episode_reward}'
            print(tmp_str)
            with open(args.full_save_path + "/" + "log.txt", '+a') as f:
                f.write(tmp_str)
                f.write("\n")
        # Save the model and training information at fixed intervals
        if exp % args.save_per_episode == 0:
            agent.save_model()
            for key in totle_info.keys():
                np.save(args.full_save_path + "/" + str(key) + "_"+args.alg + ".npy", totle_info[key])
