import pybullet as p
import pybullet_data
import time
import numpy as np

def print_joint_info(robot_id):
    """打印机器人所有关节的信息"""
    num_joints = p.getNumJoints(robot_id)
    print(f"\n机器人总关节数: {num_joints}")
    print("\n关节信息:")
    print("-" * 50)
    
    for joint_id in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_id)
        print(f"关节 ID: {joint_id}")
        print(f"关节名称: {joint_info[1].decode('utf-8')}")
        print(f"关节类型: {joint_info[2]}")
        print(f"第一个位置索引: {joint_info[3]}")
        print(f"第一个速度索引: {joint_info[4]}")
        print(f"父链接索引: {joint_info[16]}")
        print("-" * 50)

def test_robot_model():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置摄像机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=0,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0]
    )
    
    p.loadURDF("plane.urdf")
    
    p.setGravity(0, 0, -9.81)
    
    # robot = p.loadURDF("urdf/car.urdf", [0, 0, 0.1])
    robot = p.loadURDF("urdf/car.urdf", [0, 0, 0.5])
    print_joint_info(robot)

    print("按 q 键退出...")
    while p.isConnected():
        # 刷新物理引擎
        p.stepSimulation()
        time.sleep(1./240.)  # 240Hz
        
        # 获取并打印机器人姿态
        pos, orn = p.getBasePositionAndOrientation(robot)
        linear_vel, angular_vel = p.getBaseVelocity(robot)
        euler = p.getEulerFromQuaternion(orn)
        euler_deg = [e * 180/3.14159 for e in euler]
        # print(f"机器人位置: {pos}, 姿态: {orn}")
        # print(f"欧拉角 (RPY) 度数: roll={euler_deg[0]:.2f}°, pitch={euler_deg[1]:.2f}°, yaw={euler_deg[2]:.2f}°")        
        print(f"车身角速度 (rad/s): wx={angular_vel[0]:.2f}, wy={angular_vel[1]:.2f}, wz={angular_vel[2]:.2f}")
    p.disconnect()

def test_robot_movement():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置摄像机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=90,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0]
    )
    
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    robot = p.loadURDF("robot.urdf", [0, 0, 0.1])
    print_joint_info(robot)

    # 定义控制参数
    max_force = 10  # 最大力矩
    target_velocity = 5  # 目标角速度
    time_step = 0  # 用于控制周期运动

    print("按 q 键退出...")
    while p.isConnected():
        # 使用正弦函数产生周期性的速度
        velocity = target_velocity * np.sin(time_step * 0.1)
        
        # 控制左右轮子
        p.setJointMotorControl2(robot, 0, p.VELOCITY_CONTROL, 
                              targetVelocity=velocity, 
                              force=max_force)
        p.setJointMotorControl2(robot, 1, p.VELOCITY_CONTROL, 
                              targetVelocity=velocity, 
                              force=max_force)
        
        # 更新时间步
        time_step += 1
        
        # 刷新物理引擎
        p.stepSimulation()
        # time.sleep(1./240.)
        
        # 获取并打印机器人姿态
        pos, orn = p.getBasePositionAndOrientation(robot)
        print(f"机器人位置: {pos}")
        
    p.disconnect()

if __name__ == "__main__":
    test_robot_model()