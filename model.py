import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor网络（动作策略）
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic网络（价值评估）
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        features = self.feature(state)
        
        # Actor输出
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic输出
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)
        normal = Normal(action_mean, action_std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(-1)
        return action, log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        
        # PPO超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 64
        
        self.buffer = []
        
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
        
    def train(self):
        if len(self.buffer) < self.batch_size:
            return
            
        batch = self.buffer
        self.buffer = []
        
        states = torch.FloatTensor([item[0] for item in batch])
        actions = torch.FloatTensor([item[1] for item in batch])
        rewards = torch.FloatTensor([item[2] for item in batch])
        next_states = torch.FloatTensor([item[3] for item in batch])
        dones = torch.FloatTensor([item[4] for item in batch])
        old_log_probs = torch.FloatTensor([item[5] for item in batch])
        old_values = torch.FloatTensor([item[6] for item in batch])
        
        # 计算优势函数
        advantages = self._compute_gae(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.ppo_epochs):
            # 获取新的动作分布和价值估计
            action_mean, action_std, values = self.actor_critic(states)
            dist = Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(-1)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算Actor损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算Critic损失
            value_loss = 0.5 * (values.squeeze() - (rewards + self.gamma * (1 - dones) * old_values)).pow(2).mean()
            
            # 总损失
            loss = actor_loss + 0.5 * value_loss
            
            # 更新网络
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.FloatTensor(advantages)