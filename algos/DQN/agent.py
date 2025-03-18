import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math

from common.models import Qnet
from common.memories import ReplayBuffer

class Agent:
    def __init__(self, cfg):
        # 定义超参
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.action_dim = cfg.action_dim

        # e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.max_steps = cfg.max_steps

        self.policy_net = Qnet(cfg.state_shape[0], cfg.state_shape[1], cfg.action_dim).to(self.device)
        self.target_net = Qnet(cfg.state_shape[0], cfg.state_shape[1], cfg.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 

        self.memory = ReplayBuffer(cfg.buffer_size)

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
            action = self.policy_net(state).max(1)[1].item() # choose action corresponding to the maximum q value
        return action

    def update(self, transition_dict):
        # transition -> torch.tensor
        states  = torch.tensor(transition_dict['states'], device=self.device, dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'], device=self.device).unsqueeze(1)
        rewards = torch.tensor(transition_dict['rewards'], device=self.device, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(transition_dict['next_states'], device=self.device, dtype=torch.float)
        dones   = torch.tensor(transition_dict['dones'], device=self.device, dtype=torch.float).unsqueeze(1)

        # 当前状态 states , 执行动作 actions 的 Q 值
        q_values = self.policy_net(states).gather(dim=1, index=actions)
        # 下一状态 next_states 的 maxQ
        max_next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)

        # 计算损失
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones) # TD误差目标
        dqn_loss  = nn.MSELoss()(q_values, q_targets)

        # 更新 policy 网络
        self.optimizer.zero_grad() # 梯度清零
        dqn_loss.backward() # 反向传播，更新参数
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 更新 target 网络
        if self.sample_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


if __name__ == "__main__":
    pass