import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import keyboard

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, previous_tvec):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    previous_tvec - Translation vector from the previous detection

    return:
    frame - The frame with the axis drawn on it (with detected markers and axis if detected)
    new_tvec - The detected translation vector for calculating offset
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

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.04, matrix_coefficients, distortion_coefficients
            )
            new_tvec = tvec  # Store the latest translation vector

            # Calculate and print the translation difference if there was a previous tvec
            if previous_tvec is not None:
                translation_difference = new_tvec - previous_tvec
                print("Translation difference:", translation_difference.flatten())

            # Draw a square around the markers and axis
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, new_tvec


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

    detect_next = False  # Flag to control detection
    previous_tvec = None  # Initialize previous translation vector as None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Perform pose estimation only if detect_next is True
        if detect_next:
            output, new_tvec = pose_estimation(frame, aruco_dict_type, k, d, previous_tvec)
            previous_tvec = new_tvec  # Update previous_tvec with the latest translation vector
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
            time.sleep(0.1)

        if cv2.waitKey(1) == 27:
            break
        # # Check for keyboard input
        # key = cv2.waitKey(30) & 0xFF
        # if key == ord('q'):  # Press 'q' to quit
        #     break
        # elif key == 13:  # Press Enter to perform the next detection
        #     detect_next = True

    video.release()
    cv2.destroyAllWindows()
