import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import socket
import keyboard


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, previous_tvec, previous_rvec,
                    connection):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    previous_tvec - Translation vector from the previous detection
    previous_rvec - Rotation vector from the previous detection
    connection - TCP socket connection for sending data

    return:
    frame - The frame with the axis drawn on it (with detected markers and axis if detected)
    new_tvec - The detected translation vector for calculating offset
    new_rvec - The detected rotation vector for calculating rotation offset
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, cv2.aruco_dict,
        parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients
    )

    new_tvec = None  # Placeholder for current translation vector
    new_rvec = None  # Placeholder for current rotation vector

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.04, matrix_coefficients, distortion_coefficients
            )
            new_tvec = tvec  # Store the latest translation vector
            new_rvec = rvec  # Store the latest rotation vector

            # Calculate and send the translation and rotation differences if there was a previous tvec and rvec
            if previous_tvec is not None and previous_rvec is not None:
                translation_difference = (new_tvec - previous_tvec).flatten()
                print("Translation difference:", translation_difference)

                # Convert rotation vectors to rotation matrices
                rotation_matrix_current, _ = cv2.Rodrigues(rvec)
                rotation_matrix_previous, _ = cv2.Rodrigues(previous_rvec)

                # Calculate rotation difference as a rotation matrix
                rotation_difference_matrix = np.dot(rotation_matrix_previous.T, rotation_matrix_current)
                print("Rotation difference matrix:\n", rotation_difference_matrix)

                # Flatten and concatenate translation and rotation for sending
                message = ','.join(map(str, translation_difference)) + ',' + ','.join(
                    map(str, rotation_difference_matrix.flatten()))

                # Send translation and rotation difference to the client
                connection.sendall(message.encode())

            # Draw a square around the markers and axis
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, new_tvec, new_rvec


if __name__ == '__main__':
    # Argument parser for input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    ap.add_argument("-v", "--video", help="Path to the video file (optional)")
    args = vars(ap.parse_args())

    # Load camera parameters from files
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])

    # Validate ArUco tag type
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]

    # Open video stream from the default camera or file if provided
    if args["video"] is None:
        video = cv2.VideoCapture(0)  # Default to the webcam
    else:
        video = cv2.VideoCapture(args["video"])

    # Allow time for camera to initialize
    time.sleep(2.0)

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('172.26.211.137', 10000)
    sock.bind(server_address)
    sock.listen(1)
    print('Waiting for a connection...')
    connection, client_address = sock.accept()
    print(f'Connection from {client_address}')

    detect_next = False  # Flag to control detection
    previous_tvec = None  # Initialize previous translation vector as None
    previous_rvec = None  # Initialize previous rotation vector as None

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Perform pose estimation only if detect_next is True
            if detect_next:
                output, new_tvec, new_rvec = pose_estimation(frame, aruco_dict_type, k, d, previous_tvec, previous_rvec,
                                                             connection)
                previous_tvec = new_tvec  # Update previous_tvec with the latest translation vector
                previous_rvec = new_rvec  # Update previous_rvec with the latest rotation vector
                detect_next = False  # Reset flag after detection
            else:
                output = frame  # Show the original frame if not detecting

            # Display the frame
            cv2.imshow('Estimated Pose', output)

            # Check keyboard input
            if keyboard.is_pressed('q'):  # Press "q" to quit
                break
            elif keyboard.is_pressed('enter'):  # Press "Enter" for next detection
                detect_next = True
                time.sleep(0.3)

            if cv2.waitKey(1) == 27:
                break

    finally:
        # Cleanup resources
        video.release()
        cv2.destroyAllWindows()
        connection.close()
        sock.close()
