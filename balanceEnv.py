from pyexpat import model
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet
import pybullet_data
import numpy as np
import os
import time
import math

class BalanceEnv(gym.Env):
    _shared_client = None
    _client_counter = 0

    def __init__(self, render=True, 
                 max_steps=1000, 
                 shared_memory=True, 
                 urdf_path=None, 
                 model_center=None,
                 camera_distance=1.5,
                 camera_yaw=0,
                 camera_pitch=0,
                 apply_disturbances=False,  # 新增参数：是否应用扰动
                 enable_position_noise=True,
                 position_noise_range=2.5,
                 enable_orientation_noise=True, 
                 orientation_noise_range=0.1,
                 enable_velocity_noise=True,
                 velocity_noise_range=10.0,
                 enable_external_force=False,
                 force_magnitude_range=(10, 50),
                 ):
        super().__init__()

        # client
        self.current_step = 0
        self.shared_memory = shared_memory

        # robot
        self.model_center = model_center if model_center is not None else [0, 0, 0]
        self.urdf_path = urdf_path
        self.render_mode = render
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-10000.0, high=10000.0, shape=(3,), dtype=np.float32
        )

        # pybullet
        self.physics_client = None
        self.robot_id = None
        self.joint_indices = {}
        self.plane_id = None

        self.dt = 1/240  # 仿真时间步长
        self.gravity = -9.81

        # reward weights
        self.pitch_weight = 1.0
        self.roll_weight = 1.0
        self.yaw_weight = 1.0
        self.velocity_weight = 0.1
        self.action_weight = 0.01
        self.position_weight = 2.0

        self.consecutive_balanced_steps = 0
        self.balance_threshold = 0.3  # 小于这个角度被认为是平衡的
        self.was_unbalanced = False  # 上一步是否失去平衡

        # camera
        self.camera_distance = camera_distance
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch
        
        # motor
        # self.motor_config = {
        #     "flywheel1": {
        #         "max_torque": 1000.0,  # 最大力矩
        #         "dead_zone": 172.0,    # PWM死区
        #         "curve_exp": 1.2       # PWM到力矩的映射曲线指数
        #     },
        #     "flywheel2": {
        #         "max_torque": 1000.0,  # 最大力矩
        #         "dead_zone": 172.0,    # PWM死区
        #         "curve_exp": 1.2       # PWM到力矩的映射曲线指数
        #     },
        #     "forward": {
        #         "max_torque": 2000.0,  # 最大力矩
        #         "dead_zone": 100.0,    # PWM死区
        #         "curve_exp": 1.0       # PWM到力矩的映射曲线指数
        #     },
        # }

        # noise and external forces
        self.apply_disturbances = apply_disturbances
        self.enable_position_noise = enable_position_noise
        self.position_noise_range = position_noise_range
        self.enable_orientation_noise = enable_orientation_noise
        self.orientation_noise_range = orientation_noise_range
        self.enable_velocity_noise = enable_velocity_noise
        self.velocity_noise_range = velocity_noise_range
        self.enable_external_force = enable_external_force
        self.force_magnitude_range = force_magnitude_range

        # Initialize the environment
        self._initialize_client()
        self._init_env()
        self._setup_camera()

    def _initialize_client(self):
        """初始化 PyBullet 客户端"""
        if self.shared_memory and BalanceEnv._shared_client is not None:
            # 重用现有的客户端连接
            self.physics_client = BalanceEnv._shared_client
            BalanceEnv._client_counter += 1
            print(f"重用现有的 PyBullet 客户端连接，当前连接数: {BalanceEnv._client_counter}")
        else:
            # 创建新的客户端连接
            if self.render_mode:
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
                
            # 如果启用了共享内存且这是第一个连接，则保存它
            if self.shared_memory:
                BalanceEnv._shared_client = self.physics_client
                BalanceEnv._client_counter = 1
                print(f"创建新的 PyBullet 客户端连接 ID: {self.physics_client}")
    
    def _setup_camera(self):
        """设置摄像机视角"""
        if self.render:
            # 设置摄像机距离、位置和方向
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_distance,  # 距离 - 调整这个值可以拉远或拉近视角
                cameraYaw=self.camera_yaw,        # 水平旋转角度 (0-360度)
                cameraPitch=self.camera_pitch,     # 垂直俯仰角度 (-90到90度)
                cameraTargetPosition=[0, 0, 0.2],  # 摄像机目标位置 [x, y, z]
                physicsClientId=self.physics_client
            )
            
            # 设置渲染选项
            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS, 0,
                physicsClientId=self.physics_client
            )
            
            # 改变背景颜色
            p.configureDebugVisualizer(
                p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1,
                physicsClientId=self.physics_client
            )
            
            # 显示或隐藏GUI控件
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0,  # 设置为0可隐藏GUI控件
                physicsClientId=self.physics_client
            )

    def _init_env(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.dt, physicsClientId=self.physics_client)

        # 加载平面
        if self.plane_id is None:
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
            p.changeDynamics(
                self.plane_id, 
                -1, 
                lateralFriction=0.1,  # 设置为更真实的桌布摩擦系数
                physicsClientId=self.physics_client
            )

        # 加载机器人模型
        if self.robot_id is None:
            if self.urdf_path is not None:
                if os.path.exists(self.urdf_path):
                    self.robot_id = p.loadURDF(self.urdf_path, self.model_center, [0, 0, 0, 1], physicsClientId=self.physics_client)
                else:
                    raise FileNotFoundError(f"URDF 文件未找到: {self.urdf_path}")
            else:
                raise ValueError("请提供有效的 URDF 路径。")
        else:
            # 如果机器人已经加载，重置位置
            p.resetBasePositionAndOrientation(self.robot_id, self.model_center, [0, 0, 0, 1], physicsClientId=self.physics_client)
            self._get_joint_info()
            for joint in self.joint_indices.values():
                p.resetJointState(
                    self.robot_id, 
                    joint, 
                    targetValue=0, 
                    targetVelocity=0, 
                    physicsClientId=self.physics_client
                )
            
            # 稳定化机器人初始状态
            for _ in range(3):
                p.stepSimulation(physicsClientId=self.physics_client)

        # 重置步数
        self.current_step = 0

        if self.apply_disturbances:
            self._disturbances()

        print("环境初始化完成。")
    
    def _disturbances(self):
        """应用扰动"""
        if self.enable_position_noise:
            position_noise = np.random.uniform(-self.position_noise_range, self.position_noise_range, size=3)
            p.resetBasePositionAndOrientation(self.robot_id, 
                                              np.array(self.model_center) + position_noise, 
                                              [0, 0, 0, 1], 
                                              physicsClientId=self.physics_client)
        
        if self.enable_orientation_noise:
            orientation_noise = np.random.uniform(-self.orientation_noise_range, self.orientation_noise_range, size=3)
            p.resetBasePositionAndOrientation(self.robot_id, 
                                              self.model_center, 
                                              p.getQuaternionFromEuler(orientation_noise), 
                                              physicsClientId=self.physics_client)
        
        if self.enable_velocity_noise:
            velocity_noise = np.random.uniform(-self.velocity_noise_range, self.velocity_noise_range)
            for joint in self.joint_indices.values():
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=velocity_noise,
                    force=1000.0,
                    physicsClientId=self.physics_client
                )
        
        if self.enable_external_force:
            force_magnitude = np.random.uniform(*self.force_magnitude_range)
            p.applyExternalForce(
                objectUniqueId=self.robot_id,
                linkIndex=-1,
                forceObj=[force_magnitude, 0, 0],
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.physics_client
            )

    def _get_joint_info(self):
        """获取关节信息"""
        self.joint_indices = {}
        try:
            num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
            
            print(f"机器人总关节数: {num_joints}")
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                joint_name = joint_info[1].decode('utf-8')
                self.joint_indices[joint_name] = i
                print(f"关节 {i}: {joint_name}")
                
            print(f"找到的关节: {list(self.joint_indices.keys())}")
        except Exception as e:
            print(f"获取关节信息错误: {e}")

    def _get_observation(self):
        """获取当前状态的观测值"""
        joint_states = p.getJointStates(self.robot_id, list(self.joint_indices.values()), physicsClientId=self.physics_client)
        # joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        
        # 获取机器人姿态
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        base_euler = p.getEulerFromQuaternion(base_orientation)
        _, angular_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        
        roll, pitch, yaw = base_euler
        roll_vel, pitch_vel, yaw_vel = angular_vel
        forward_vel, fly1_vel, fly2_vel = joint_velocities

        pitch = math.degrees(pitch)
        roll = math.degrees(roll)
        yaw = math.degrees(yaw)

        # 观测值
        observation = np.array([
            pitch, roll, yaw,
            pitch_vel, roll_vel, yaw_vel,
            forward_vel, fly1_vel, fly2_vel
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, observation, action):
        """计算奖励函数"""
        pitch, roll, yaw, pitch_vel, roll_vel, yaw_vel, forward_vel, fly1_vel, fly2_vel = observation
        
        # 保持直立的奖励（pitch接近0）
        pitch_reward = -self.pitch_weight * abs(pitch)
        roll_reward = -self.roll_weight * abs(roll)
        yaw_reward = -self.yaw_weight * abs(yaw)  # 如果需要yaw奖励，可以取消注释
        
        # 惩罚过大的角速度
        # velocity_penalty = -self.velocity_weight * (abs(roll_vel) + abs(pitch_vel))
        
        # # 惩罚过大的动作
        # action_penalty = -self.action_weight * abs(action)

        # 位置偏离惩罚 - 使用平方或指数函数让惩罚随距离增加而迅速增大
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        horizontal_dist = np.sqrt(current_pos[0]**2 + current_pos[1]**2)
        position_penalty = -self.position_weight * horizontal_dist**2

        # 平衡持续时间奖励
        is_balanced = abs(pitch) < self.balance_threshold and abs(roll) < self.balance_threshold
        if is_balanced:
            self.consecutive_balanced_steps += 1
            balance_time_reward = 0.1 * np.log(1 + self.consecutive_balanced_steps * 0.1)
        else:
            self.consecutive_balanced_steps = 0
            balance_time_reward = 0

        # 恢复平衡奖励
        recovery_reward = 0
        if is_balanced and self.was_unbalanced:
            recovery_reward = 2.0
        self.was_unbalanced = not is_balanced

        # 生存奖励
        survival_reward = 0.1  # 每个时间步的生存奖励
        
        total_reward = pitch_reward + survival_reward + position_penalty + balance_time_reward + recovery_reward + roll_reward
        
        return total_reward
    
    def _is_terminated(self, observation):
        """检查是否终止"""
        pitch, roll, yaw, pitch_vel, roll_vel, yaw_vel, forward_vel, fly1_vel, fly2_vel = observation
        
        # 如果倾斜角度过大，终止
        if abs(pitch) > 10 or abs(roll) > 10:  # 60度
            return True
            
        # 如果角速度过大，终止
        # if abs(roll_vel) > 10 or abs(pitch_vel) > 10:
        #     return True
            
        try:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
            if abs(pos[2]) > 0.5 or abs(pos[0]) > 0.5 or abs(pos[2]) > 0.5: 
                return True
        except:
            return True
            
        return False
    
    def _pwm_to_torque(self, pwm_value, max_torque=1000.0, dead_zone=172.0, curve_exp=1.2):
        """
        将单个关节的PWM值转换为力矩
        
        参数:
            pwm_value: 单个关节的PWM值 (-10000到10000)
            joint_name: 关节名称，用于获取特定配置（如果有）
            max_torque: 最大力矩（默认值）
            dead_zone: PWM死区（默认值）
            curve_exp: PWM到力矩的映射曲线指数（默认值）
            
        返回:
            计算得到的力矩值
        """
        # 确保PWM在有效范围内
        pwm_value = np.clip(pwm_value, -10000.0, 10000.0)
        
        # PWM归一化和应用死区
        norm_pwm = pwm_value / 10000.0
        norm_dead_zone = dead_zone / 10000.0
        
        # 应用死区
        if abs(norm_pwm) < norm_dead_zone:
            adj_pwm = 0.0
        else:
            # 处理死区外的值
            sign = np.sign(norm_pwm)
            adj_pwm = sign * (abs(norm_pwm) - norm_dead_zone) / (1.0 - norm_dead_zone)
        
        # 计算力矩（非线性映射）
        torque = max_torque * np.sign(adj_pwm) * np.power(np.abs(adj_pwm), curve_exp)
        
        return torque
    
    def _pwm_to_torque_with_gearbox(self, pwm_value, base_torque = 93.1, gear_ratio=4.4, gear_efficiency=0.85, dead_zone=100.0, curve_exp=1.1):
        """
        将PWM值转换为RC-370S减速电机的力矩，考虑减速比
        
        参数:
            pwm_value: PWM值 (-10000到10000)
            joint_name: 关节名称，用于获取特定配置
            
        返回:
            计算得到的力矩值 (N·m)
        """
        # 确保PWM在有效范围内
        pwm_value = np.clip(pwm_value, -10000.0, 10000.0)
        
        # 计算减速后的最大力矩 (N·mm -> N·m)
        max_torque = (base_torque * gear_ratio * gear_efficiency) / 1000.0
        
        # PWM归一化和应用死区
        norm_pwm = pwm_value / 10000.0
        norm_dead_zone = dead_zone / 10000.0
        
        # 应用死区
        if abs(norm_pwm) < norm_dead_zone:
            adj_pwm = 0.0
        else:
            sign = np.sign(norm_pwm)
            adj_pwm = sign * (abs(norm_pwm) - norm_dead_zone) / (1.0 - norm_dead_zone)
        
        # 计算力矩
        torque = max_torque * np.sign(adj_pwm) * np.power(abs(adj_pwm), curve_exp)
        
        return torque
    
    def _joint_control(self, action):
        if hasattr(self, 'joint_indices') and self.joint_indices:
            if 'wheel_joint' in self.joint_indices:
                joint_idx = self.joint_indices['wheel_joint']
                p.setJointMotorControl2(
                    self.robot_id, 
                    joint_idx,
                    p.TORQUE_CONTROL,
                    # force=self._pwm_to_torque_with_gearbox(action[0]),
                    force = 10,
                    physicsClientId=self.physics_client
                )
            # print(f"设置 wheel_joint 力矩: {self._pwm_to_torque_with_gearbox(action[0])}")
            # 两个飞轮
            if 'flywheel1_joint' in self.joint_indices:
                joint_idx = self.joint_indices['flywheel1_joint']
                p.setJointMotorControl2(
                    self.robot_id, 
                    joint_idx,
                    p.TORQUE_CONTROL,
                    force=self._pwm_to_torque(action[1]),
                    physicsClientId=self.physics_client
                )
            if 'flywheel2_joint' in self.joint_indices:
                joint_idx = self.joint_indices['flywheel2_joint']
                p.setJointMotorControl2(
                    self.robot_id, 
                    joint_idx,
                    p.TORQUE_CONTROL,
                    force=self._pwm_to_torque(action[2]),
                    physicsClientId=self.physics_client
                )
            
    def step(self, action):
        self.current_step += 1
        if self.physics_client is None:
            raise RuntimeError("PyBullet 客户端未初始化，请先调用 _initialize_client() 方法。")
        
        # 设置关节控制
        self._joint_control(action)

        # 执行一步仿真
        for _ in range(2):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(0.01)  # 确保每步仿真有足够的时间

        # 获取当前状态的观测值
        observation = self._get_observation()

        # 计算奖励
        reward = self._calculate_reward(observation, action)

        # 检查是否终止
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_steps
            
        # 返回观测值、奖励、终止标志和额外信息
        info = {
            "current_step": self.current_step,
            "terminated": terminated,
        }

        return observation, reward, False, truncated, info
    
    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
        else:
            seed = np.random.randint(0, 2**32 - 1)
        return [seed]

    def reset(self,*, seed=None, options=None):
        """重置环境状态"""
        if self.physics_client is None:
            raise RuntimeError("PyBullet 客户端未初始化，请先调用 _initialize_client() 方法。")
        
        if seed is not None:
            np.random.seed(seed)
        # 重置环境
        self._init_env()
        info = {}
        # 返回初始观测

        self._setup_camera()
        return self._get_observation(), info
    
    def render(self):
        """渲染环境"""
        if self.render_mode:
            time.sleep(self.dt)

    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            # 只有在使用共享连接且是最后一个环境时才断开连接
            if self.shared_memory and self.physics_client == BalanceEnv._shared_client:
                BalanceEnv._client_counter -= 1
                if BalanceEnv._client_counter <= 0:
                    try:
                        p.disconnect(self.physics_client)
                        BalanceEnv._shared_client = None
                        BalanceEnv._client_counter = 0
                        print("关闭 PyBullet 共享连接")
                    except:
                        pass
            # 如果不是共享连接，直接断开
            elif not self.shared_memory:
                try:
                    p.disconnect(self.physics_client)
                    print(f"关闭 PyBullet 独立连接 ID: {self.physics_client}")
                except:
                    pass
            
            self.physics_client = None