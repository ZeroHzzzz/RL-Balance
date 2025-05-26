from robotEnv import BalanceRobotEnv
from model import PPOAgent
import torch
import numpy as np
import time

# 创建环境和智能体
env = BalanceRobotEnv()
state_dim = 8
action_dim = 1
agent = PPOAgent(state_dim, action_dim)

# 训练参数
num_episodes = 1000
max_steps = 1000

# 训练循环
for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")
    state = env.reset()[0]
    total_reward = 0
    
    for step in range(max_steps):
        # 选择动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = agent.actor_critic.get_action(state_tensor)
        action = action.detach().numpy()[0]
        
        # 执行动作
        next_state, reward, done, _, _ = env.step(action)
        print(next_state)
        time.sleep(0.01)
        # 存储transition
        agent.store_transition(
            state, action, reward, next_state, done, 
            log_prob.item(), value.item()
        )
        
        total_reward += reward
        state = next_state
        
        # 训练
        if len(agent.buffer) >= agent.batch_size:
            agent.train()
        
        if done:
            break
    
    # 打印训练信息
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")