#!/usr/bin/env python

# Author: Anton Deguet
# Date: 2015-02-22

# (C) Copyright 2015-2020 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

# Start a single arm using
# > rosrun dvrk_robot dvrk_console_json -j <console-file>

# To communicate with the arm using ROS topics, see the python based example dvrk_arm_test.py:
# > rosrun dvrk_python dvrk_arm_test.py <arm-name>

import dvrk
import math
import sys
import rospy
import numpy as np
import PyKDL
import time
import argparse
import socket
import time
import keyboard
import pybullet as p

# from Control_platform_test import Cont_platform

if sys.version_info.major < 3:
    input = raw_input

# print with node id
def print_id(message):
    print('%s -> %s' % (rospy.get_caller_id(), message))

# example of application using arm.py
class example_application:

    # configuration
    def configure(self, robot_name, expected_interval):
        print_id('configuring dvrk_psm_test for %s' % robot_name)
        self.expected_interval = expected_interval
        self.arm = dvrk.psm(arm_name = robot_name,
                            expected_interval = expected_interval)

    # homing example
    def home(self):
        print_id('starting enable')
        if not self.arm.enable(10):
            sys.exit('failed to enable within 10 seconds')
        print_id('starting home')
        if not self.arm.home(10):
            sys.exit('failed to home within 10 seconds')
        # # get current joints just to set size
        # print_id('move to starting position')
        # goal = numpy.copy(self.arm.setpoint_jp())
        # # go to zero position, make sure 3rd joint is past cannula
        # goal.fill(0)
        # goal[2] = 0.12
        # self.arm.move_jp(goal).wait()



    # goal jaw control example




    def run_jaw_servo(self, angle):
        print('aaaaaaaaaaaaaangle', angle)
        start_angle = math.radians(angle)
        print('ssssssssssstart angle', start_angle)
        self.arm.jaw.servo_jp(np.array(start_angle))
        rospy.sleep(self.expected_interval)

    # def run_jaw_servo(self, angle):
    #     try:
    #         # start_angle = math.radians(angle)
    #         # self.arm.jaw.servo_jp(np.array(0.15))
    #         # print(start_angle)
    #         self.arm.jaw.servo_jp(np.array(angle))
    #         # rospy.sleep(self.expected_interval)
    #     except:
    #         print('unable to  run_jaw_servo')

    def current_gripper(self):
        try:
            joints = np.copy(self.arm.jaw.setpoint_jp())
            return joints
        except:
            print('unable to get setpoint_cp()')

    # def run_move_cp(self, ax,ay,az):   #we added az here
    #     print("initial movement of assisted arm")
    #     start_cp = PyKDL.Frame()
    #     start_cp.p = self.arm.setpoint_cp().p
    #     start_cp.M = self.arm.setpoint_cp().M

    #     goal = PyKDL.Frame()
    #     goal.p = self.arm.setpoint_cp().p
    #     goal.M = self.arm.setpoint_cp().M

    #     goal.p[0] = start_cp.p[0] + ay
    #     goal.p[1] = start_cp.p[1] + ax
    #     goal.p[2] = start_cp.p[2] + az

    #     handle = self.arm.move_cp(goal).wait(is_busy=True)
    #     print ('move cp time = ')

    def run_servo_cp(self, ax,ay,az):
        time1 = time.time()
        start_cp = PyKDL.Frame()
        try:
            start_cp.p = self.arm.setpoint_cp().p
            start_cp.M = self.arm.setpoint_cp().M

            goal = PyKDL.Frame()
            goal.p = self.arm.setpoint_cp().p
            goal.M = self.arm.setpoint_cp().M

            # amplitude = LC4 #-0.00001  # 2 centimeters
            goal.p[0] = start_cp.p[0] + ay
            goal.p[1] = start_cp.p[1] + ax
            goal.p[2] = start_cp.p[2] + az

            # handle = self.move_cp(goal)
            handle = self.arm.servo_cp(goal) ##
            # print('This is goal: ', goal)
            rospy.sleep(self.expected_interval)
            # handle.wait() #0.001
            time2 = time.time()
            move_time = time2 - time1
            print ('move cp time = ', move_time)
        
        except:
            print('unable to get setpoint_jp in namespace PSM2')





    # def run_servo_cp(self, ax,ay,az, au,av,aw):
    #     time1 = time.time()
    #     start_cp = PyKDL.Frame()
    #     try:
    #         start_cp.p = self.arm.setpoint_cp().p
    #         start_cp.M = self.arm.setpoint_cp().M

    #         goal = PyKDL.Frame()
    #         goal.p = self.arm.setpoint_cp().p
    #         goal.M = self.arm.setpoint_cp().M

    #         # amplitude = LC4 #-0.00001  # 2 centimeters
    #         # goal.p[0] = start_cp.p[0] + ax
    #         # goal.p[1] = start_cp.p[1] - ay
    #         # goal.p[2] = start_cp.p[2] - az


    #         ax = np.clip(ax, -0.001, 0.001)
    #         ay = np.clip(ay, -0.001, 0.001)
    #         az = np.clip(az, -0.001, 0.001)


    #         goal.p[0] = start_cp.p[0] +0.139*ax +0.9914*ay - 0.0032*az
    #         goal.p[1] = start_cp.p[1] -0.9909*ax +0.1307*ay - 0.0306*az
    #         goal.p[2] = start_cp.p[2] +0.0299*ax + 0.0072*ay - 0.9995*az

    #         goal.M.DoRotX(math.pi * -aw)
    #         goal.M.DoRotY(math.pi * -au)
    #         goal.M.DoRotZ(math.pi * av)

    #         # goal.M[0] = start_cp.M[0] + au
    #         # goal.M[1] = start_cp.M[1] + av
    #         # goal.M[2] = start_cp.M[2] + aw
    #         # print('THIS is M: ', goal.M)

    #         # handle = self.move_cp(goal)
    #         handle = self.arm.servo_cp(goal) ##
    #         # print('This is goal: ', goal)
    #         rospy.sleep(self.expected_interval)
    #         # handle.wait() #0.001
    #         time2 = time.time()
    #         move_time = time2 - time1
    #         # print ('move cp time = ', move_time)
    #     except:
    #         print('unable to get setpoint_jp in namespace PSM2')

    def current_state(self):
        time1 = time.time()
        try:
            start_cp = PyKDL.Frame()
            start_cp.p = self.arm.setpoint_cp().p
            start_cp.M = self.arm.setpoint_cp().M

            rospy.sleep(self.expected_interval)
            time2 = time.time()
            move_time = time2 - time1

            return start_cp.p * 1000
            # return start_cp.p
        except:
            print('unable to current_state')


    def get_joints(self):
        init_joint_position = np.array([0,0,0,0,0,0])
        try:
            current_joint_position = np.copy(self.arm.setpoint_jp())

            return current_joint_position
        except:
            print('unable to get setpoint_jp in namespace PSM2')

    
    def return_to_init_servo_jp(self, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7):

        # init_joint_position = np.array([-1.02304754, -0.04029722,  0.18462101, 2.9, -0.39433294, -0.61295881])
        # init_joint_position = np.array([-0.7653, -0.12142281, 0.1746797, 1.717785, 0.62062, -1.24844])
        #                                   base                    prismatic                         tip


        try:
            init_joint_position = np.array([joint_1, joint_2, joint_3, joint_4, joint_5, joint_6])


            current_joint_position = np.copy(self.arm.setpoint_jp())
            goal = np.copy(current_joint_position)

            dist_joint = init_joint_position - current_joint_position
            TOL = 0.025

            start = rospy.Time.now()

            # goal[5] = current_joint_position[5] + 0.1
            if dist_joint[5]>TOL:
                goal[5] = current_joint_position[5] + 0.01
                # print("doing this1")
            if dist_joint[5]<TOL:
                goal[5] = current_joint_position[5] - 0.01
                # print("doing this1")

            if dist_joint[4]>TOL:
                goal[4] = current_joint_position[4] + 0.01
                # print("doing this3")
            if dist_joint[4]<TOL:
                goal[4] = current_joint_position[4] - 0.01
                # print("doing this3")

            # print(init_joint_position[3], '   ', current_joint_position[3], '   ', dist_joint[3])
            if dist_joint[3]>TOL:
                goal[3] = current_joint_position[3] + 0.01
            if dist_joint[3]<TOL:
                goal[3] = current_joint_position[3] - 0.01

            if dist_joint[2]>TOL*0.01:    ###Prismatic joint
                goal[2] = current_joint_position[2] + 0.0002
            if dist_joint[2]<TOL*0.01:
                goal[2] = current_joint_position[2] - 0.0002

            if dist_joint[1]>TOL*0.1:    ###Prismatic joint
                goal[1] = current_joint_position[1] + 0.001
            if dist_joint[1]<TOL*0.1:
                goal[1] = current_joint_position[1] - 0.001

            if dist_joint[0]>TOL*0.1:    ###Prismatic joint
                goal[0] = current_joint_position[0] + 0.001
                # print("doing this2")
            if dist_joint[0]<TOL*0.1:
                goal[0] = current_joint_position[0] - 0.001
                # print("doing this2")

            # goal = numpy.copy(current_joint_position)

            # self.run_jaw_servo(0.15)
            # time.sleep(5)
            self.arm.servo_jp(goal)
            # self.arm.jaw.servo_jp(0.2)
            # self.arm.jaw.open(0.2)
            
            # amplitude = math.radians(30.0)
            # durition = 5
            # samples = int(durition / self.expected_interval)
            # for i in range (samples*4):
            #     goal1 = start_angle + amplitude*(math.cos(i*math.radians(360.0)/samples) - 1.0)
            #     self.arm.jaw.servo_jp(np.array(goal1))
            #     rospy.sleep(self.expected_interval)

            rospy.sleep(self.expected_interval)
            actual_duration = rospy.Time.now() - start
            # print('jp GOAL: ', goal)
            # print("DISTANCE (NOT YET)::: ", dist_joint)
            goal_reached = 0
            if all(abs(dist_joint) < np.array([TOL,TOL,TOL,0.1,0.1,0.1])):    #[TOL,TOL,TOL,0.1,0.1,0.1]
                            #                   0   1   2   3   4       5
                goal_reached = 1
                # print("DISTANCE::: ", dist_joint)

            


            return goal_reached
        

        except:
            print('unable to return_to_init_servo_jp')

    
active_TCPIP = 1
# joint_1 = 0
# joint_2 = 0
# joint_3 = 0  #0.05
# joint_4 = 0
# joint_5 = 0
# joint_6 = 0  #0.05
# joint_7 = 0
if __name__ == '__main__':
    ##### HOST = '129.31.147.72'
    # if active_TCPIP == 1:
    #     HOST = '172.30.28.72'
    #     PORT = 30000
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #     s.bind((HOST, PORT))
    #     s.listen(10) # socket queue of 10
    #     conn, addr = s.accept()

    application = example_application()
    application.configure('PSM1', 0.014)
    application2 =example_application()
    application2.configure('PSM2', 0.014)

    
    application.home()
    application2.home()

    current_joints_1 = application.get_joints()
    current_joints_2 = application2.get_joints()

    print("current_joints_1: ",current_joints_1)
    # print("current_joints_2: ",current_joints_2)


    offset_x = 0.005
    offset_y = 0.005 # 5mm
    offset_z = 0.005
    returned_frame_psm1 = application.run_servo_cp( offset_x,offset_y,offset_z)
    # returned_frame_psm2 = application2.run_servo_cp( offset_x,offset_y,offset_z)
    print("current_joints_1_offset: ",current_joints_1)
    # print("current_joints_2_offset: ",current_joints_2)
    # returned_frame_psm1 = application.run_servo_cp_tele(b*omega_z_step, -b*omega_x_step, b*omega_y_step, 0, 0, 0, sigma_rel)
    # returned_frame_psm2 = application2.run_servo_cp_tele(d*omega_z_step_R,-d*omega_x_step_R, d*omega_y_step_R, 0, 0, 0, sigma_rel_R)


    