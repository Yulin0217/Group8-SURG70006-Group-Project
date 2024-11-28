'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import keyboard
import socket


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    ids - Detected marker IDs
    tvecs - Translation vectors for each detected marker
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, cv2.aruco_dict, parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients
    )

    tvecs = []
    rvecs = []

    if len(corners) > 0:
        for i in range(len(ids)):
            # Estimate pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.04, matrix_coefficients, distortion_coefficients
            )
            rvecs.append(rvec[0])
            tvecs.append(tvec[0])

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, ids, tvecs


if __name__ == '__main__':

    ################### Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('172.26.143.151', 12000)
    sock.bind(server_address)
    sock.listen(1)
    print('Waiting for a connection...')
    connection, client_address = sock.accept()
    print(f'Connection from {client_address}')
    ##############################################

    # Load ArUco dictionary type
    aruco_dict_type = ARUCO_DICT["DICT_5X5_100"]

    # Paths to calibration files
    calibration_matrix_path = "calibration_matrix.npy"
    distortion_coefficients_path = "distortion_coefficients.npy"

    # Load camera calibration data
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # Start video stream
    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    detect_next = False  # Flag to control detection

    while True:
        ret, frame = video.read()
        if not ret:
            break

        translation_matrix = []  # Reset translation matrix for each frame

        if detect_next:
            # Pose estimation
            output_frame, ids, tvecs = pose_esitmation(frame, aruco_dict_type, k, d)

            if ids is not None and len(ids) > 1:
                # Sort markers by their IDs for consistent calculation
                sorted_indices = np.argsort(ids.flatten())
                ids = ids.flatten()[sorted_indices]
                tvecs = np.array(tvecs)[sorted_indices]

                # Calculate translation differences
                for i in range(len(tvecs) - 1):
                    diff = tvecs[i + 1] - tvecs[i]
                    translation_matrix.append(diff.flatten())  # Append to the matrix
                    print(f"Translation difference between Marker {ids[i]} and Marker {ids[i+1]}: {diff}")

                # Convert translation matrix to string and send over TCP/IP
                translation_matrix_str = np.array(translation_matrix).flatten().tolist()
                data_to_send = str(translation_matrix_str).encode('utf-8')
                connection.sendall(data_to_send)
            else:
                print("Not enough markers detected for comparison.")

            detect_next = False

        else:
            output_frame = frame

        # Display the frame
        cv2.imshow('Estimated Pose', output_frame)

        if keyboard.is_pressed('q'):  # Press "q" to quit
            break
        elif keyboard.is_pressed('enter'):  # Press "Enter" for next detection
            detect_next = True
            time.sleep(0.3)

        if cv2.waitKey(1) == 27:
            break

    video.release()
    connection.close()
    cv2.destroyAllWindows()
