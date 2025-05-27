import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import argparse
from balanceEnv import BalanceEnv
import time
import pybullet as p
import test

def quick_test():
    """快速测试环境是否正常工作"""
    print("进行快速环境测试...")
    
    env = BalanceEnv(urdf_path="car.urdf", model_center=np.array([0, 0, 0.075]), shared_memory=True, camera_distance=1.0, camera_pitch=0, camera_yaw=0.0)
    # test_env = BalanceEnv(render_mode='hunman', urdf_path="robot.urdf", model_center=np.array([0.3, 0.3, 0.3]), shared_memory=False)
    
    # try:
    obs = env.reset()
    # test_env.reset()
    
    print(f"初始观测: {obs}")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    
    # 运行几步  
    for i in range(1000):
        time.sleep(0.2)  # 确保渲染有时间显示
        # action = env.action_space.sample()
        action = np.array([10000, 0.0, 0.0])  # 使用零动作测试
        obs, reward, terminated, truncated, info = env.step(action)
        
        # print(obs)
        wheel_idx = env.joint_indices['wheel_joint']
        wheel_state = p.getJointState(env.robot_id, wheel_idx, physicsClientId=env.physics_client)
        wheel_velocity = wheel_state[1]
        applied_torque = wheel_state[3]
        
        print(f"Wheel velocity: {wheel_velocity}, Applied torque: {applied_torque}")
    
        # print(f"步骤 {i+1}: 奖励 = {float(reward):.3f}, 终止 = {terminated}")  # 确保reward是标量            
        if terminated or truncated:
            print("回合结束，重置环境")
            obs = env.reset()
        
    print("环境测试成功！")
        
    # finally:
    #     env.close()


if __name__ == "__main__":
    # 如果没有命令行参数，运行快速测试然后训练
    quick_test()
