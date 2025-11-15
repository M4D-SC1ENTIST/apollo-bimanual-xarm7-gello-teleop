import time
from dataclasses import dataclass

import numpy as np
from gello.robots.xarm_robot import XArmRobot
from gello.robots.robot import BimanualRobot
from gello.zmq_core.robot_node import ZMQServerRobot

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
                 viewpoint_arm_start_joints_state: list = None):
        
        if arm_to_use == "left" or arm_to_use == "both":
            self.left_arm = XArmRobot(ip=left_arm_ip)

        if arm_to_use == "right" or arm_to_use == "both":
            self.right_arm = XArmRobot(ip=right_arm_ip)

        # Reset left arm
        if arm_to_use == "left" or arm_to_use == "both":
            ret = self.left_arm.robot.set_servo_angle(angle=left_arm_start_joints_state[0:7], speed=0.3, wait=True, is_radian=True)
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
            ret = self.right_arm.robot.set_servo_angle(angle=right_arm_start_joints_state[0:7], speed=0.3, wait=True, is_radian=True)
            ret = self.right_arm.command_joint_state(np.array(right_arm_start_joints_state))

            if ret in [1, 9]:
                self.right_arm._clear_error_states()
            self.right_arm.robot.set_gripper_position(right_arm_start_joints_state[7], wait=True)
            
            self.right_arm.robot.set_mode(1)
            self.right_arm.robot.set_state(state=0)
            self.right_arm.robot.clean_error()
        
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


if __name__ == "__main__":
    xarm7_robots_manager = xArm7RobotsManager()