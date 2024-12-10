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
import time
import rospy
import numpy
import PyKDL
import argparse
import socket
import crtk
import keyboard


# print with node id
def print_id(message):
    print('%s -> %s' % (rospy.get_caller_id(), message))

# example of application using arm.py
class example_application:

    # configuration
    # def configure(self, ral, robot_name, expected_interval):
    #     print_id('configuring dvrk_arm_test for %s' % robot_name)
    #     self.expected_interval = expected_interval
    #     self.arm = dvrk.arm(arm_name = robot_name,
    #                         expected_interval = expected_interval)
    
    # configuration
    def configure(self, ral, arm_name, expected_interval):
        print('configuring dvrk_arm_test for {}'.format(arm_name))
        self.ral = ral
        self.expected_interval = expected_interval
        self.arm = dvrk.arm(ral = ral,
                            arm_name = arm_name,
                            expected_interval = expected_interval)

    # homing example
    # def home(self):
    #     print_id('starting enable')
    #     if not self.arm.enable(10):
    #         sys.exit('failed to enable within 10 seconds')
    #     print_id('starting home')
    #     if not self.arm.home(10):
    #         sys.exit('failed to home within 10 seconds')
    #     # get current joints just to set size
    #     print_id('move to starting position')
    #     goal = numpy.copy(self.arm.setpoint_jp())
    #     # go to zero position, for PSM and ECM make sure 3rd joint is past cannula
    #     goal.fill(0)
    #     if ((self.arm.name() == 'PSM1') or (self.arm.name() == 'PSM2')
    #         or (self.arm.name() == 'PSM3') or (self.arm.name() == 'ECM')):
    #         goal[2] = 0.12
    #     # move and wait
    #     print_id('moving to starting position')
    #     self.arm.move_jp(goal).wait()
    #     # try to move again to make sure waiting is working fine, i.e. not blocking
    #     print_id('testing move to current position')
    #     move_handle = self.arm.move_jp(goal)
    #     time.sleep(1.0) # add some artificial latency on this side
    #     move_handle.wait()
    #     print_id('home complete')

    def home(self):
        print_id('starting enable')
        if not self.arm.enable(10):
            sys.exit('failed to enable within 10 seconds')
        print_id('starting home')
        if not self.arm.home(10):
            sys.exit('failed to home within 10 seconds')

    # get methods
    def run_get(self):
        [p, v, e, t] = self.arm.measured_js()
        d = self.arm.measured_jp()
        [d, t] = self.arm.measured_jp(extra = True)
        d = self.arm.measured_jv()
        [d, t] = self.arm.measured_jv(extra = True)
        d = self.arm.measured_jf()
        [d, t] = self.arm.measured_jf(extra = True)
        d = self.arm.measured_cp()
        [d, t] = self.arm.measured_cp(extra = True)
        d = self.arm.local.measured_cp()
        [d, t] = self.arm.local.measured_cp(extra = True)
        d = self.arm.measured_cv()
        [d, t] = self.arm.measured_cv(extra = True)
        d = self.arm.body.measured_cf()
        [d, t] = self.arm.body.measured_cf(extra = True)
        d = self.arm.spatial.measured_cf()
        [d, t] = self.arm.spatial.measured_cf(extra = True)

        [p, v, e, t] = self.arm.setpoint_js()
        d = self.arm.setpoint_jp()
        [d, t] = self.arm.setpoint_jp(extra = True)
        d = self.arm.setpoint_jv()
        [d, t] = self.arm.setpoint_jv(extra = True)
        d = self.arm.setpoint_jf()
        [d, t] = self.arm.setpoint_jf(extra = True)
        d = self.arm.setpoint_cp()
        [d, t] = self.arm.setpoint_cp(extra = True)
        d = self.arm.local.setpoint_cp()
        [d, t] = self.arm.local.setpoint_cp(extra = True)

    # direct joint control example
    def run_servo_jp(self):
        print_id('starting servo_jp')
        # get current position
        initial_joint_position = numpy.copy(self.arm.setpoint_jp())
        print_id('testing direct joint position for 2 joints out of %i' % initial_joint_position.size)
        amplitude = math.radians(5.0) # +/- 5 degrees
        duration = 5  # seconds
        samples = duration / self.expected_interval
        # create a new goal starting with current position
        goal = numpy.copy(initial_joint_position)
        start = rospy.Time.now()
        for i in xrange(int(samples)):
            goal[0] = initial_joint_position[0] + amplitude *  (1.0 - math.cos(i * math.radians(360.0) / samples))
            goal[1] = initial_joint_position[1] + amplitude *  (1.0 - math.cos(i * math.radians(360.0) / samples))
            self.arm.servo_jp(goal)
            rospy.sleep(self.expected_interval)
        actual_duration = rospy.Time.now() - start
        print_id('servo_jp complete in %2.2f seconds (expected %2.2f)' % (actual_duration.to_sec(), duration))

    # goal joint control example
    def run_move_jp(self):
        print_id('starting move_jp')
        # get current position
        initial_joint_position = numpy.copy(self.arm.setpoint_jp())
        print_id('testing goal joint position for 2 joints out of %i' % initial_joint_position.size)
        amplitude = math.radians(10.0)
        # create a new goal starting with current position
        goal = numpy.copy(initial_joint_position)
        # first motion
        goal[0] = initial_joint_position[0] + amplitude
        goal[1] = initial_joint_position[1] - amplitude
        self.arm.move_jp(goal).wait()
        # second motion
        goal[0] = initial_joint_position[0] - amplitude
        goal[1] = initial_joint_position[1] + amplitude
        self.arm.move_jp(goal).wait()
        # back to initial position
        self.arm.move_jp(initial_joint_position).wait()
        print_id('move_jp complete')

    # utility to position tool/camera deep enough before cartesian examples
    def prepare_cartesian(self):
        # make sure the camera is past the cannula and tool vertical
        goal = numpy.copy(self.arm.setpoint_jp())
        if ((self.arm.name() == 'PSM1') or (self.arm.name() == 'PSM2')
            or (self.arm.name() == 'PSM3') or (self.arm.name() == 'ECM')):
            # set in position joint mode
            goal[0] = 0.0
            goal[1] = 0.0
            goal[2] = 0.12
            goal[3] = 0.0
            self.arm.move_jp(goal).wait()

    # direct cartesian control example
    def run_servo_cp(self, offset_x, offset_y, offset_z):
        # print_id('starting servo_cp')
        # self.prepare_cartesian()

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M
        # motion parameters
        # amplitude = 0.005 # 4 cm total
        # duration = 1  # 5 seconds
        # samples = duration / self.expected_interval

        # print('start',initial_cartesian_position)

        # start = rospy.Time.now()
        # for i in xrange(int(samples)):
        goal.p[0] =  initial_cartesian_position.p[0] + offset_y #*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
        goal.p[1] =  initial_cartesian_position.p[1] + offset_x #*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
        goal.p[2] =  initial_cartesian_position.p[2] + offset_z #*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
            # if i == 180:
            #     print_id("target(step{180}):")
            #     print_id("rotation: {}".format(goal.M))
            #     print_id("translation {}:".format(goal.p))

            
            
        self.arm.servo_cp(goal)
            # check error on kinematics, compare to desired on arm.
            # to test tracking error we would compare to
            # current_position
        # setpoint_cp = self.arm.setpoint_cp()
        # errorX = goal.p[0] - setpoint_cp.p[0]
        # errorY = goal.p[1] - setpoint_cp.p[1]
        # errorZ = goal.p[2] - setpoint_cp.p[2]
        # error = math.sqrt(errorX * errorX + errorY * errorY + errorZ * errorZ)
        # if error > 0.002: # 2 mm
        #     print_id('Inverse kinematic error in position [%i]: %s (might be due to latency)' % (i, error))
        # rospy.sleep(self.expected_interval)
        # actual_duration = rospy.Time.now() - start
        # print_id('servo_cp complete in %2.2f seconds (expected %2.2f)' % (actual_duration.to_sec(), duration))

        # print_id('')
        # after_cartesian_position = PyKDL.Frame()
        # after_cartesian_position.p = self.arm.setpoint_cp().p
        # after_cartesian_position.M = self.arm.setpoint_cp().M
        # print('after',after_cartesian_position)
        # print_id('move_cp complete')

    # direct cartesian control example
    def run_move_cp(self, offset_x, offset_y, offset_z):
        print_id('starting move_cp')
        # self.prepare_cartesian() 

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M

        # # motion parameters
        # amplitude = 0.01 # 1 mm

        # first motion
        goal.p[0] =  initial_cartesian_position.p[0] + offset_y
        goal.p[1] =  initial_cartesian_position.p[1] + offset_x
        goal.p[2] =  initial_cartesian_position.p[2] + offset_z

        # print('start',initial_cartesian_position)

        # self.arm.move_cp(goal).wait() ###################################################################
        # # second motion
        # goal.p[0] =  initial_cartesian_position.p[0] + amplitude
        # goal.p[1] =  initial_cartesian_position.p[1]
        # self.arm.move_cp(goal).wait()
        # # back to initial position
        # goal.p[0] =  initial_cartesian_position.p[0]
        # goal.p[1] =  initial_cartesian_position.p[1]
        # self.arm.move_cp(goal).wait()
        # # first motion
        # goal.p[0] =  initial_cartesian_position.p[0]
        # goal.p[1] =  initial_cartesian_position.p[1] - amplitude
        # self.arm.move_cp(goal).wait()
        # # second motion
        # goal.p[0] =  initial_cartesian_position.p[0]
        # goal.p[1] =  initial_cartesian_position.p[1] + amplitude
        # self.arm.move_cp(goal).wait()
        # # back to initial position
        # goal.p[0] =  initial_cartesian_position.p[0]
        # goal.p[1] =  initial_cartesian_position.p[1]
        # self.arm.move_cp(goal).wait()

        # print_id('')
        # after_cartesian_position = PyKDL.Frame()
        # after_cartesian_position.p = self.arm.setpoint_cp().p
        # after_cartesian_position.M = self.arm.setpoint_cp().M
        # print('after',after_cartesian_position)
        # # print('Cartesian Offset', (after_cartesian_position - initial_cartesian_position))
        # print_id('move_cp complete')


    def run_move_cp_rotation(self, offset_x, offset_y, offset_z, rotation_offset):
        print_id('starting move_cp')
        self.prepare_cartesian()

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M

        # first motion
        goal.p[0] =  initial_cartesian_position.p[0] + offset_y
        goal.p[1] =  initial_cartesian_position.p[1] + offset_x
        goal.p[2] =  initial_cartesian_position.p[2] + offset_z

        rotation_matrix_offset = PyKDL.Rotation(
            rotation_offset[0, 0], rotation_offset[0, 1], rotation_offset[0, 2],
            rotation_offset[1, 0], rotation_offset[1, 1], rotation_offset[1, 2],
            rotation_offset[2, 0], rotation_offset[2, 1], rotation_offset[2, 2],
        )

        goal.M = initial_cartesian_position.M * rotation_matrix_offset

        self.arm.move_cp(goal).wait()
    
    def run_servo_cp_steps(self, offset_x, offset_y, offset_z, steps, dt):
        print_id('starting servo_cp')
        # self.prepare_cartesian()
        for i in range(1, steps):
        # create a new goal starting with current position
            initial_cartesian_position = PyKDL.Frame()
            initial_cartesian_position.p = self.arm.setpoint_cp().p
            initial_cartesian_position.M = self.arm.setpoint_cp().M
            goal = PyKDL.Frame()
            goal.p = self.arm.setpoint_cp().p
            goal.M = self.arm.setpoint_cp().M

            # print('start',initial_cartesian_position)

        
            goal.p[0] =  initial_cartesian_position.p[0] + offset_y / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
            goal.p[1] =  initial_cartesian_position.p[1] + offset_x / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
            goal.p[2] =  initial_cartesian_position.p[2] + offset_z / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))       
            self.arm.servo_cp(goal)     
            time.sleep(dt)

        # # self.arm.servo_cp(goal)
        # print_id('')
        # after_cartesian_position = PyKDL.Frame()
        # after_cartesian_position.p = self.arm.setpoint_cp().p
        # after_cartesian_position.M = self.arm.setpoint_cp().M
        # print('after',after_cartesian_position)
        # print_id('move_cp complete')
        
    def run_move_cp_steps(self, offset_x, offset_y, offset_z, steps, dt):
        print_id('starting servo_cp')
        # self.prepare_cartesian()
        for i in range(1, steps):
            # create a new goal starting with current position
            initial_cartesian_position = PyKDL.Frame()
            initial_cartesian_position.p = self.arm.setpoint_cp().p
            initial_cartesian_position.M = self.arm.setpoint_cp().M
            goal = PyKDL.Frame()
            goal.p = self.arm.setpoint_cp().p
            goal.M = self.arm.setpoint_cp().M

            # print('start',initial_cartesian_position)

            goal.p[0] =  initial_cartesian_position.p[0] + offset_x / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
            goal.p[1] =  initial_cartesian_position.p[1] + offset_y / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
            goal.p[2] =  initial_cartesian_position.p[2] + offset_z / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))       
            self.arm.move_cp(goal)     
            time.sleep(dt)

    def run_set_point(self, x, y, z, steps, dt):
        print_id('starting move_cp')
        # self.prepare_cartesian() 

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        
        offset_x = x - initial_cartesian_position.p[0]
        offset_y = y - initial_cartesian_position.p[1]
        offset_z = z - initial_cartesian_position.p[2]

        # goal = PyKDL.Frame()
        # goal.p = self.arm.setpoint_cp().p
        # goal.M = self.arm.setpoint_cp().M

        for i in range(1, steps):
            # create a new goal starting with current position
            initial_cartesian_position = PyKDL.Frame()
            initial_cartesian_position.p = self.arm.setpoint_cp().p
            initial_cartesian_position.M = self.arm.setpoint_cp().M
            goal = PyKDL.Frame()
            goal.p = self.arm.setpoint_cp().p
            goal.M = self.arm.setpoint_cp().M

            # print('start',initial_cartesian_position)

            goal.p[0] =  initial_cartesian_position.p[0] + offset_x / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
            goal.p[1] =  initial_cartesian_position.p[1] + offset_y / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))
            goal.p[2] =  initial_cartesian_position.p[2] + offset_z / steps#*  (1.0 - math.cos(i/2 * math.radians(360.0) / samples))       
            self.arm.move_cp(goal)     
            time.sleep(dt)


    def run_read_pose(self):
        # print_id('starting move_cp')
        # self.prepare_cartesian() 

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        # goal = PyKDL.Frame()
        # goal.p = self.arm.setpoint_cp().p
        # goal.M = self.arm.setpoint_cp().M

        print('start',initial_cartesian_position)

        return initial_cartesian_position.p

        # # motion parameters
        # amplitude = 0.01 # 1 mm

        # # first motion
        # goal.p[0] =  initial_cartesian_position.p[0] + offset_y
        # goal.p[1] =  initial_cartesian_position.p[1] + offset_x
        # goal.p[2] =  initial_cartesian_position.p[2] + offset_z


    # main method
    def run(self):
        self.home()
        self.run_get()
        # self.run_servo_jp()
        # self.run_move_jp()
        # self.run_servo_cp()
        self.run_move_cp()

if __name__ == '__main__':

    ral = crtk.ral('dvrk_arm_test')
    application = example_application()
    application.configure(ral,'PSM1',0.01)
    application.home()
 

    # ##################     Create a TCP/IP socket   ####################
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the server's address and port
    server_address = ('172.26.143.151', 12000)
    # server_address = ('172.26.143.151', 12000)
    print(f'connecting to {server_address[0]} port {server_address[1]}', file=sys.stderr)
    sock.connect(server_address)

    steps = 30
    dt = 0.05
    buffer = ""
    detect_next = True

    # initial_cartesian = application.run_read_pose()
    # try:                                                                                                                                                                                                                                                                 
    #     while True:
    #         data = sock.recv(1024)  # Receive data from the server
    #         if not data:
    #             print("Disconnection", file=sys.stderr)
    #             break

    #         if detect_next:
    #             buffer += data.decode('utf-8')  # Decode the received data
    #             while '\n' in buffer:
    #                 line, buffer = buffer.split('\n', 1)  # Split the buffer into lines
    #                 try:
    #                     # Parse the string to a 3x4 matrix (12 values expected)
    #                     matrix_values = [float(value) for value in line.split(',')]
    #                     if len(matrix_values) != 12:
    #                         raise ValueError("Expected 12 values for a 3x4 matrix")

    #                     # Convert to a 3x4 matrix and extract the last column
    #                     matrix = numpy.array(matrix_values).reshape(3, 4)
    #                     translation_matrix = matrix[:, 3]
    #                 except ValueError as e:
    #                     print(f"Error: {line} ({e})", file=sys.stderr)
    #                     continue

    #                 # Translation matrix contains the offsets in x, y, z
    #                 offset_x = translation_matrix[0]
    #                 offset_y = translation_matrix[1]
    #                 offset_z = translation_matrix[2]

    #                 # Control the dVRK robot using the extracted offsets
    #                 # application.run_servo_cp_steps(offset_x, offset_y, offset_z, steps, dt)

    #                 print(f"Received translation matrix: {translation_matrix}", file=sys.stderr)
                    # detect_next = False

            # if keyboard.is_pressed('q'):  # Exit if "q" is pressed
            #     break
            # elif keyboard.is_pressed('enter'):  # Detect next if "Enter" is pressed
            #     detect_next = True
            #     time.sleep(0.3)

    # ################################################
    # try:
    #     while True:
    #         data = sock.recv(1024)
    #         if not data:
    #             print("Disconnection", file=sys.stderr)
    #             break
            
    #         if detect_next:
    #             buffer += data.decode('utf-8')
    #             while '\n' in buffer:
    #                 line, buffer = buffer.split('\n', 1)
    #                 try:
    #                     displacement = [float(value) for value in line.split(',')]
    #                 except ValueError:
    #                     print(f"Error: {line}", file=sys.stderr)
    #                     continue

    #                 offset_x = displacement[0]
    #                 offset_y = displacement[1]
    #                 offset_z = displacement[2]

    #                 # for i in range(steps):
    #                 application.run_move_cp_steps(offset_x, offset_y, offset_z, steps, dt)
    #                     # time.sleep(0.025)

    #                 print(f"Received offset: {displacement}", file=sys.stderr)
                    # detect_next = False
            
            # if keyboard.is_pressed('q'):
            #     break
            # elif keyboard.is_pressed('enter'):
            #     detect_next = True
            #     time.sleep(0.3)
##################################################################
    

    ################            1. Test with one image pose             ################
    try:
        while True:
            # 从服务器接收数据（缓冲区大小设置为 4096 字节）
            data = sock.recv(4096)

            if not data:
                print("No data received. Closing connection.")
                break

            # 解码接收的数据
            received_str = data.decode('utf-8')
            print(f"Raw received data: {received_str}")

            # 解析数据为 NumPy 数组
            try:
                # 将字符串转换为 Python 列表
                translation_list = eval(received_str)

                # print('Tran list: ', translation_list)

                # 将列表转换为 2D NumPy 矩阵，每行包含一个平移向量
                translation_matrix = numpy.array(translation_list).reshape(-1, 3)
                print(f"Translation Matrix:\n{translation_matrix}")

                for i in range(len(translation_matrix)):
                    displacement = translation_matrix[i]
                    # if i ==0:
                    #     offset_x = displacement[0] - 0.03
                    #     offset_y = displacement[1] - 0.03
                    # else:
                    #     offset_x = displacement[0] 

                    # 分解位移向量
                    offset_x = displacement[0]
                    offset_y = displacement[1]
                    offset_z = displacement[2]

                    # for i in range(steps):
                    application.run_move_cp_steps(offset_x, offset_y, offset_z, steps, dt)
                    pose = application.run_read_pose()
                    # print('Pose', pose)

                    # print(f"Received offset: {displacement}", file=sys.stderr)

                
            except Exception as e:
                print(f"Error parsing received data: {e}")
                continue
#  # 遍历每个位移向量并控制 dVRK
#                 # for displacement in translation_matrix:
#                 for i in range(len(translation_matrix)):
#                     displacement = translation_matrix[i]
#                     if i ==0:
#                         offset_x = displacement[0] - 0.03
#                         offset_y = displacement[1] - 0.03
#                     else:
#                         offset_x = displacement[0] 
#                         offset_y = displacement[1]

#                     # 分解位移向量
#                     # offset_x = displacement[0] 
#                     # offset_y = displacement[1]
#                     offset_z = displacement[2]

#                     # for i in range(steps):
#                     application.run_move_cp_steps(offset_x, offset_y, offset_z, steps, dt)
#                     pose = application.run_read_pose()
#                     # print('Pose', pose)

#                     # print(f"Received offset: {displacement}", file=sys.stderr)

                
#             except Exception as e:
#                 print(f"Error parsing received data: {e}")
#                 continue

    except socket.error as e:
        print(f"wrong: {e}", file=sys.stderr)
    finally:
        print("Close", file=sys.stderr)
        sock.close()
    
    # application.run_set_point(initial_cartesian[0], initial_cartesian[1], initial_cartesian[2], steps, dt)
 







    # offset_x = 0.0
    # offset_y = -0.0015
    # offset_z = 0.


    # # returned_frame_psm1 = application.run_move_cp(offset_x, offset_y, offset_z)
    # for i in range(1, 20):
    #     application.run_servo_cp(offset_x, offset_y, offset_z)
    #     time.sleep(0.1)


    