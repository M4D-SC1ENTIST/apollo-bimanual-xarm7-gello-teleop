import time
from dataclasses import dataclass

import numpy as np
from gello.robots.xarm_robot import XArmRobot
from gello.robots.robot import BimanualRobot
from gello.zmq_core.robot_node import ZMQServerRobot

from xarm.wrapper import XArmAPI

class xArm7RobotsManager:
    def __init__(self,
                 arm_to_use: str = "right", # "left", "right", "both"
                 left_arm_ip: str = "192.168.3.214",   
                 right_arm_ip: str = "192.168.1.236",
                 viewpoint_arm_ip: str = "192.168.2.219",
                 hostname: str = "127.0.0.1",
                 robot_server_port: int = 6001,
                 left_arm_start_joints_state: list = [1.62448565, -0.84829138, -2.4190877, 1.40052446, -0.63813601, 1.26400017, -0.69796126, 0.0],
                 right_arm_start_joints_state: list = [1.009359, 0.704097, 0.208621, 1.537049, 6.211088, 1.013946, 1.191903, 0.00375],
                 viewpoint_arm_start_joints_state: list = None,
                 reset_joint_speed: float = 0.5): # First element is linear motor, the following 7 are joints
        
        if arm_to_use == "left" or arm_to_use == "both":
            self.left_arm = XArmRobot(ip=left_arm_ip)

        if arm_to_use == "right" or arm_to_use == "both":
            self.right_arm = XArmRobot(ip=right_arm_ip)

        # Reset left arm
        if arm_to_use == "left" or arm_to_use == "both":
            ret = self.left_arm.robot.set_servo_angle(angle=left_arm_start_joints_state[0:7], speed=reset_joint_speed, wait=True, is_radian=True)
            ret = self.left_arm.command_joint_state(np.array(left_arm_start_joints_state))
            if ret in [1, 9]:
                self.left_arm._clear_error_states()
            self.left_arm.robot.set_gripper_position(left_arm_start_joints_state[7], wait=True)
            self.left_arm.robot.set_mode(1)
            self.left_arm.robot.set_state(state=0)
            self.left_arm.robot.clean_error()

        #self.right_arm.robot.set_mode(6)
        #self.right_arm.robot.set_state(state=0)
        #time.sleep(1)

        # Reset right arm
        if arm_to_use == "right" or arm_to_use == "both":
            ret = self.right_arm.robot.set_servo_angle(angle=right_arm_start_joints_state[0:7], speed=reset_joint_speed, wait=True, is_radian=True)
            ret = self.right_arm.command_joint_state(np.array(right_arm_start_joints_state))

            if ret in [1, 9]:
                self.right_arm._clear_error_states()
            self.right_arm.robot.set_gripper_position(right_arm_start_joints_state[7], wait=True)
            self.right_arm.robot.set_mode(1)
            self.right_arm.robot.set_state(state=0)
            self.right_arm.robot.clean_error()

        if viewpoint_arm_start_joints_state is not None:
            print("SETTING VIEWPOINT ARM")
            self.viewpoint_arm = XArmAPI(viewpoint_arm_ip, is_radian=True)
            self.viewpoint_arm.connect()

            self.clean_viewpoint_arm_error_states()
            
            
            # Set viewpoint arm to intermediate position
            intermediate_joints_state_1 = [0.0, -1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 0.0]
            intermediate_joints_state_2 = [3.141592653589793, -1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 0.0]
            ret = self.viewpoint_arm.set_servo_angle(angle=intermediate_joints_state_1, speed=reset_joint_speed, wait=True, is_radian=True)
            if ret in [1, 9]:
                self.viewpoint_arm.clean_error()
            ret = self.viewpoint_arm.set_servo_angle(angle=intermediate_joints_state_2, speed=reset_joint_speed, wait=True, is_radian=True)
            if ret in [1, 9]:
                self.viewpoint_arm.clean_error()

            # Reset viewpoint arm
            ret = self.viewpoint_arm.set_servo_angle(angle=viewpoint_arm_start_joints_state[1:8], speed=reset_joint_speed, wait=True, is_radian=True)
            if ret in [1, 9]:
                self.viewpoint_arm.clean_error()
            
            res = self.viewpoint_arm.set_linear_motor_enable(True)
            print("LINEAR MOTOR ENABLE RESULT: ", res)

            res = self.viewpoint_arm.set_linear_motor_back_origin()
            print("LINEAR MOTOR BACK ORIGIN RESULT: ", res)
            

            print("LINEAR MOTOR ENABLE OR NOT: ", self.viewpoint_arm.get_linear_motor_is_enabled())
            print("LINEAR MOTOR ON ZERO: ", self.viewpoint_arm.get_linear_motor_on_zero())
            self.check_and_fix_viewpoint_linear_motor_error()
            
            # Set linear motor position
            res = self.viewpoint_arm.set_linear_motor_pos(viewpoint_arm_start_joints_state[0] * 1000, speed=0.2 * 1000, wait=False)

            print("LINEAR MOTOR SET POSITION RESULT: ", res)
        
        # Start server
        if arm_to_use == "both":
            self.robot = BimanualRobot(self.left_arm, self.right_arm)
        elif arm_to_use == "left":
            self.robot = self.left_arm
        elif arm_to_use == "right":
            self.robot = self.right_arm

        self.server = ZMQServerRobot(self.robot, port=robot_server_port, host=hostname)
        print(f"Starting robot server on port {robot_server_port}")
        self.server.serve()
    
    def clean_viewpoint_arm_error_states(self):
        self.viewpoint_arm.connect()
        self.viewpoint_arm.clean_error()
        self.viewpoint_arm.clean_warn()
        self.viewpoint_arm.clean_linear_motor_error()
        self.viewpoint_arm.set_collision_sensitivity(0)
        self.viewpoint_arm.set_self_collision_detection(0)
        self.viewpoint_arm.motion_enable(True)
        time.sleep(1)
        self.viewpoint_arm.set_mode(0)
        self.viewpoint_arm.set_state(0)
    
    def check_and_fix_viewpoint_linear_motor_error(self):
        res = self.viewpoint_arm.get_linear_motor_error()
        print("LINEAR MOTOR ERROR: ", res)
        if res[1] != 0:
            self.viewpoint_arm.clean_linear_motor_error()


if __name__ == "__main__":
    xarm7_robots_manager = xArm7RobotsManager()