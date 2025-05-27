import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import time
from datetime import datetime
from balanceEnv import BalanceEnv

def create_env(render=True, model_center=None, urdf_path="car.urdf", shared_memory=False):
    """创建平衡车环境"""
    env = BalanceEnv(
        render=render,
        urdf_path=urdf_path,
        model_center=model_center if model_center is not None else np.array([0, 0, 0.075]),
        max_steps=1000,
        shared_memory=shared_memory,
        camera_distance=1.0,
        camera_pitch=0,
        camera_yaw=0.0,
        apply_disturbances=True,
    )
    return Monitor(env)  # 使用Monitor包装器以记录训练指标

def train_agent(algorithm='PPO', total_timesteps=100000, save_path='models', 
                learning_rate=3e-4, render_training=False, eval_freq=5000,
                  continue_training=False, model_path=None):
    
    # 创建日期时间标记，确保每次训练有唯一标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{algorithm}_{timestamp}"
    
    # 创建保存目录
    model_dir = os.path.join(save_path, run_id)
    log_dir = os.path.join("tensorboard_logs", run_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建训练环境
    env = create_env(render=render_training)
    
    # 创建评估环境
    eval_env = create_env(render=False)
    
    # 设置回调函数
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=5000, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(model_dir, 'best'),
        log_path=os.path.join(model_dir, 'logs'),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_train_callback,
        n_eval_episodes=5
    )
    
    model = None

    if continue_training and model_path:

        print(f"加载模型进行继续训练: {model_path}")

        try:
            # 检测模型类型
            if 'ppo' in model_path.lower():
                model = PPO.load(model_path, env=env)
                print("加载PPO模型成功")
            elif 'sac' in model_path.lower():
                model = SAC.load(model_path, env=env)
                print("加载SAC模型成功")
            elif 'td3' in model_path.lower():
                model = TD3.load(model_path, env=env)
                print("加载TD3模型成功")
            else:
                # 尝试自动检测
                try:
                    model = PPO.load(model_path, env=env)
                    print("自动检测: 加载PPO模型成功")
                except:
                    try:
                        model = SAC.load(model_path, env=env)
                        print("自动检测: 加载SAC模型成功")
                    except:
                        model = TD3.load(model_path, env=env)
                        print("自动检测: 加载TD3模型成功")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将从头开始训练新模型")
            continue_training = False  # 重置为非继续训练模式
    
    if not continue_training or model is None:
        print(f"创建新的{algorithm}模型")

        # 选择并配置算法
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=learning_rate,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto'
        )
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=learning_rate,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    # 打印训练信息
    print(f"开始使用 {algorithm} 算法训练平衡车...")
    print(f"总训练步数: {total_timesteps}")
    print(f"模型ID: {run_id}")
    print(f"模型保存路径: {model_dir}")
    print(f"学习率: {learning_rate}")
    
    # 记录训练开始时间
    start_time = time.time()

    try:
        # 开始训练
        model.learn(
            total_timesteps=total_timesteps, 
            callback=eval_callback,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, 'final_model')
        model.save(final_model_path)
        print(f"训练完成！最终模型已保存到 {final_model_path}")
        
    except KeyboardInterrupt:
        print("训练被用户中断")
        # 保存中断时的模型
        interrupted_path = os.path.join(model_dir, 'interrupted_model')
        model.save(interrupted_path)
        print(f"已保存中断时的模型到 {interrupted_path}")
    
    finally:
        # 计算训练时间
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"训练用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        
        # 关闭环境
        env.close()
        eval_env.close()
    
    return model, model_dir

if __name__ == "__main__":    
    default_config = {
        "algorithm": "PPO",
        "total_timesteps": 1000000000,
        "learning_rate": 3e-4,
        "render_training": True,
        "continue_training": True,
        "model_path": "best_model.zip"
    }

    model, model_dir = train_agent(
        algorithm=default_config["algorithm"],
        total_timesteps=default_config["total_timesteps"],
        learning_rate=default_config["learning_rate"],
        render_training=default_config["render_training"],
        eval_freq=5000,  # 固定评估频率
        continue_training=default_config["continue_training"],
        model_path=default_config["model_path"]
    )