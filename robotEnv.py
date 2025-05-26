import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class BalanceRobotEnv(gym.Env):
    def __init__(self, render_mode=True):
        if render_mode:
            self.physics_client = p.connect(p.GUI)
            # 设置相机
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=90,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 加载地面和机器人
        self.plane = p.loadURDF("plane.urdf")

        # 设置地面的动力学参数
        p.changeDynamics(
            self.plane,
            -1,  # -1 表示基础链接
            lateralFriction=1.0,        # 侧向摩擦系数
            rollingFriction=0.0,        # 滚动摩擦系数
            spinningFriction=0.0,       # 自旋摩擦系数
            restitution=0.5,            # 弹性系数
            contactStiffness=1e5,       # 接触刚度
            contactDamping=1            # 接触阻尼
        )

        self.robot = p.loadURDF("robot.urdf", [0, 0, 0])
        
        # 获取电机ID
        self.left_wheel = 0  # 左轮关节ID
        self.right_wheel = 1  # 右轮关节ID
        
        # 设置电机参数
        self.max_velocity = 10  # 最大角速度（rad/s）
        self.max_torque = 1.0   # 最大力矩（N·m）
        
        # 定义动作空间：两个电机的PWM值，范围[-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        for wheel in [self.left_wheel, self.right_wheel]:
            p.changeDynamics(
                self.robot,
                wheel,
                lateralFriction=1.0,  # 增加侧向摩擦
                spinningFriction=0.1,  # 添加旋转摩擦
                rollingFriction=0.1,   # 添加滚动摩擦
                jointDamping=0.1       # 添加关节阻尼
            )
        # 定义观察空间：
        # [pitch角度, pitch角速度, 
        #  x方向加速度, y方向加速度, z方向加速度,
        #  左轮速度, 右轮速度]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        # 重置机器人位置和姿态
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.1], [0, 0, 0, 1])
        p.resetJointState(self.robot, self.left_wheel, 0)
        p.resetJointState(self.robot, self.right_wheel, 0)
        
        return self._get_observation(), {}

    def step(self, action):
        # 将[-1, 1]的PWM值转换为实际的电机力矩
        left_torque = action[0] * self.max_torque
        # right_torque = action[1] * self.max_torque
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        if pos[2] > 0.5:
            print("机器人位置过高，重置位置")

    
        # 设置电机力矩
        # p.setJointMotorControl2(self.robot, self.left_wheel, 
        #                       p.TORQUE_CONTROL,
        #                       force=left_torque)
        
        # p.setJointMotorControl2(self.robot, self.right_wheel, 
        #                       p.TORQUE_CONTROL,
        #                       force=left_torque)
        p.setJointMotorControl2(self.robot, self.left_wheel, 
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_torque,
                                force=self.max_torque)
        p.setJointMotorControl2(self.robot, self.right_wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_torque,
                                force=self.max_torque)
        
        # 模拟一步
        p.stepSimulation()
        time.sleep(1./240.)

        # 获取新的状态
        state = self._get_observation()
        
        # 计算奖励
        reward = self._compute_reward(state)
        
        # 检查是否结束
        done = self._check_termination(state)
        
        return state, reward, done, False, {}

    def _get_observation(self):
        # 获取底盘的姿态
        _, orientation = p.getBasePositionAndOrientation(self.robot)
        
        pitch = p.getEulerFromQuaternion(orientation)[0]  # 获取pitch角度
        roll = p.getEulerFromQuaternion(orientation)[1]  # 获取roll角度
        yaw = p.getEulerFromQuaternion(orientation)[2]    # 获取yaw角度

        # 获取IMU数据（角速度和加速度）
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)
        
        # 获取轮子速度
        left_vel = p.getJointState(self.robot, self.left_wheel)[1]
        right_vel = p.getJointState(self.robot, self.right_wheel)[1]
        
        return np.array([
            pitch,                # pitch角度
            roll,                 # roll角度
            yaw,                  # yaw角度
            angular_vel[0],      # pitch角速度
            angular_vel[1],      # roll角速度
            angular_vel[2],      # yaw角速度
            left_vel,           # 左轮速度
            right_vel           # 右轮速度
        ])

    def _compute_reward(self, state):
        # 奖励函数：保持直立（pitch接近0）并保持适当的移动速度
        pitch = state[0]
        roll = state[1]
        yaw = state[2]

        pitch_rate = state[3]
        roll_rate = state[4]
        yaw_rate = state[5]

        left_wheel_speed = state[6]
        right_wheel_speed = state[7]

        # 角度惩罚：pitch角度越大，惩罚越大
        pitch_angle_penalty = -abs(pitch)
        roll_angle_penalty = -abs(roll)
        yaw_angle_penalty = -abs(yaw)

        # 角速度惩罚：防止剧烈摆动
        pitch_rate_penalty = -abs(pitch_rate)
        roll_rate_penalty = -abs(roll_rate)
        yaw_rate_penalty = -abs(yaw_rate)

        # 轮子速度惩罚：保持适当的移动速度
        left_wheel_speed_penalty = -abs(left_wheel_speed)
        right_wheel_speed_penalty = -abs(right_wheel_speed)
        
        return pitch_angle_penalty + roll_angle_penalty + yaw_angle_penalty + \
               pitch_rate_penalty + roll_rate_penalty + yaw_rate_penalty + \
               left_wheel_speed_penalty + right_wheel_speed_penalty

    def _check_termination(self, state):
        # 如果倾角过大，则终止
        pitch = state[0]
        return abs(pitch) > np.pi/4  # 45度

    def close(self):
        p.disconnect()