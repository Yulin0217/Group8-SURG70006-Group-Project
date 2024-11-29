import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import socket
import keyboard

camera_matrix = np.array([[910.0994, 0., 656.3054],
                          [0., 910.1795, 359.7876],
                          [0., 0., 1.]])
dist_coeffs = np.array([0.0539, -0.1899, -0.0013, -0.0026, 0])

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, initial_tvec, initial_rvec,
                    connection):
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
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

            # Set initial vectors if they haven't been set
            if initial_tvec[0] is None and initial_rvec[0] is None:
                initial_tvec[0] = new_tvec
                initial_rvec[0] = new_rvec

            # Calculate and send the translation and rotation differences if initial vectors exist
            if initial_tvec[0] is not None and initial_rvec[0] is not None:
                translation_difference = (new_tvec - initial_tvec[0]).flatten()
                print("Translation difference:", translation_difference)

                # Convert rotation vectors to rotation matrices
                rotation_matrix_current, _ = cv2.Rodrigues(rvec)
                rotation_matrix_initial, _ = cv2.Rodrigues(initial_rvec[0])

                # Calculate rotation difference as a rotation matrix
                rotation_difference_matrix = np.dot(rotation_matrix_initial.T, rotation_matrix_current)
                print("Rotation difference matrix:\n", rotation_difference_matrix)

                # Round values to two decimal places
                rounded_translation_difference = [round(val, 2) for val in translation_difference]
                rounded_rotation_difference = [round(val, 2) for val in rotation_difference_matrix.flatten()]

                # Construct the message with rounded values
                message = ','.join(map(str, rounded_translation_difference)) + ',' + ','.join(map(str, rounded_rotation_difference)) + '\n'

                # Send translation and rotation difference to the client
                connection.sendall(message.encode())

                time.sleep(0.5)

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
    server_address = ('172.26.40.196', 12000)
    sock.bind(server_address)
    sock.listen(1)
    print('Waiting for a connection...')
    connection, client_address = sock.accept()
    print(f'Connection from {client_address}')

    detect_next = True  # Flag to control detection
    initial_tvec = [None]  # List wrapper to pass by reference
    initial_rvec = [None]  # List wrapper to pass by reference

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Perform pose estimation only if detect_next is True
            if detect_next:

                output, new_tvec, new_rvec = pose_estimation(frame, aruco_dict_type, k, d, initial_tvec, initial_rvec,
                                                             connection)

            # Display the frame
            cv2.imshow('Estimated Pose', output)

            key = cv2.waitKey(500) & 0xFF
            if key == ord('q'):
                break

    finally:
        # Cleanup resources
        video.release()
        cv2.destroyAllWindows()
        connection.close()
        sock.close()
