import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
from pathlib import Path

from common.models import Qnet
from common.memories import ReplayBuffer

class Agent:
    def __init__(self, cfg):
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.state_shape = cfg.state_shape
        self.action_dim = cfg.action_dim
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.buffer_size = cfg.buffer_size
        
        # e-greedy parameter
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end   = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.max_steps     = cfg.max_steps

        self.policy_net = Qnet(self.state_shape[0], self.state_shape[1], self.action_dim).to(self.device)
        self.target_net = Qnet(self.state_shape[0], self.state_shape[1], self.action_dim).to(self.device)
        # 将policy_net的参数赋值给target_net
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        # or self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(self.buffer_size)

    def sample_action(self, state):
        """ sample action with e-greedy policy """
        self.sample_count += 1
        self.epsilon = max(self.epsilon_end, \
                        self.epsilon_start - (self.epsilon_start-self.epsilon_end) * self.sample_count / self.epsilon_decay)
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                action = self.policy_net(state).argmax(dim=1).item()
        else:
            action = np.random.randint(self.action_dim)
        return action
    
    def predict_action(self, state):
        """ predict action """
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            action = self.policy_net(state).max(dim=1)[1].item()
        return action

    def update(self, transition_dict):
        """ difference with DQN """
        # transiton -> torch.tensor
        states  = torch.tensor(transition_dict['states'], device=self.device, dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'], device=self.device).unsqueeze(dim=1)
        rewards = torch.tensor(transition_dict['rewards'], device=self.device, dtype=torch.float).unsqueeze(dim=1)
        next_states = torch.tensor(transition_dict['next_states'], device=self.device, dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], device=self.device, dtype=torch.float).unsqueeze(dim=1)

        # 当前状态 states , 执行动作 actions 的 Q 值
        q_values = self.policy_net(states).gather(dim=1, index=actions)
        """ 区别于DQN """
        # 下一状态 next_states, out of policy_net and target_net
        next_policy_values = self.policy_net(next_states)
        next_target_values = self.target_net(next_states)
        # 从 next_policy_values 中选取 Q 值最大的动作；在next_target_value对该动作的 Q 值，作为 Q(s`, a`)的估计
        # 对应 DQN 的 self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        next_target_q_values = next_target_values.gather(dim=1, index=torch.max(next_policy_values, dim=1)[1].unsqueeze(dim=1))

        # 计算损失
        q_targets = rewards + self.gamma * next_target_q_values * (1-dones)
        doubledqn_loss = nn.MSELoss()(q_values, q_targets)

        # 更新网络
        self.optimizer.zero_grad()
        doubledqn_loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters(): 
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  

        if self.sample_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict()) 

    def save_model(self, fpath):
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pth")

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pth"))  
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)  