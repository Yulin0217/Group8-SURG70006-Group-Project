#!/usr/bin/env python3
import sys
# print(sys.version)
import os
import glob
import time
import yaml



#This code is a direct edit from node_dvrk_track_backup2-ORIGINAL_no-edit.py and called v3 since there was a ...real_v2.py
#This vision works along side the new dcrk PC by send the TCPIP information to the new script on the new dvrk PC

import rospkg
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.io import savemat

import cv2 as cv
import numpy as np
import subprocess as sp
######!/home/hs3/anaconda3/envs/rostest/bin python3
import multiprocessing as mp
import signal
import socket
from utils import Charuco_detect
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
# print(sys.version)
def resize_img(img):
    height, width, _ = img.shape
    im_odd = img[::2, :]  # odd rows
    width = int(im_odd.shape[1] / 2)
    height = int(im_odd.shape[0])
    dim = (width, height)
    
    # resize image
    resized = cv.resize(im_odd, dim, interpolation = cv.INTER_AREA)
    # print("height   ", height)
    # print("width    ", width)
    # print("resized  ", resized)

    return resized
def unpack_homo(homo):
    R = homo[:,:3][:3]
    t = homo[:,-1][:3]
    return R,t

def save_pose_mat(im_path, mat):
    filename = '{}.mat'.format(im_path)
    savemat(filename, {'data': mat})

def save_pose_np(im_path, mat):
    filename = '{}.npy'.format(im_path)
    np.save(filename, mat)

def get_3d_obj_coordinate(d, PATTERN_SHAPE):
    """ This function gets the 3D coordinates of the points

        hexagonal grid pattern:

                       d
                      |-|
            x   x   x   x -
              x   x   x   - d
            x   x   x   x
              x   x   x
            x   x   x   x
              x   x   x

        d = DIST_BETWEEN_BLOBS
    """
    pts3d = []
    z = 0 # all points in the same plane
    for i in range(PATTERN_SHAPE[1]):
        y_shift = d
        if i % 2 == 0:
            y_shift = 0
        for j in range(PATTERN_SHAPE[0]):
            """ i =
                    0   2   4   6
                      1   3   5
                    x   x   x   x
                      x   x   x
                    x   x   x   x
                      x   x   x
            """
            x = i * d
            y = j * d * 2.0
            pts3d.append([[x, y + y_shift, z]])
    pts3d = np.asarray(pts3d, dtype=float)
    return pts3d


def create_blob_detector():
    """ This function creates the blob detector with the same settings as Jian's paper """
    # TODO: hardcoded values
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 20
    params.maxThreshold = 220
    params.minDistBetweenBlobs = 5
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 1000
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.95
    params.filterByColor = True
    params.blobColor = 0
    params.minRepeatability = 2
    detector = cv.SimpleBlobDetector_create(params)
    return detector


def is_path_file(string):
    """ This function checks if the file path exists """
    if os.path.isfile(string):
        return string
    else:
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)
        return None


def load_yaml_data(path):
    """ This function loads data from a yaml file """
    if is_path_file(path):
        with open(path) as f_tmp:
            return yaml.load(f_tmp, Loader=yaml.FullLoader)


def load_camera_parameters(cam_calib_file):
    """ This function loads the intrinsic and distortion camera parameters """
    if is_path_file(cam_calib_file):
        cam_calib_data = load_yaml_data(cam_calib_file)
        cam_matrix = cam_calib_data['camera_matrix']['data']
        cam_matrix = np.reshape(cam_matrix, (3, 3))
        dist_coeff = cam_calib_data['dist_coeff']['data']
        dist_coeff = np.array(dist_coeff)
    return cam_matrix, dist_coeff


def load_image_and_undistort_it(im_path, cam_matrix, dist_coeff):
    """ This function load an image and undistorts it """
    if is_path_file(im_path):
        im = cv.imread(im_path, -1)
        im = cv.undistort(im, cam_matrix, dist_coeff)
    return im

def show_blob_detector_result(im, blob_detector):
    ## Find blob keypoints
    keypoints = blob_detector.detect(im) # mask_green_closed (try with that instead)
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Keypoints", img_with_keypoints)
    cv.waitKey(1)


def get_2d_coordinates(im, PATTERN_SHAPE, blob_detector):
    ret, corners = cv.findCirclesGrid(im, PATTERN_SHAPE, flags=(cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING), blobDetector=blob_detector)
    if ret:
        return corners
    return None


def show_axis(im, rvecs, tvecs, cam_matrix, dist_coeff, length):
    #print(cam_matrix)
    #print(np.transpose(tvecs))
    axis = np.float32([[0, 0, 0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    #print(axis)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam_matrix, dist_coeff)
    imgpts = imgpts.astype(int)
    frame_centre = tuple(imgpts[0].ravel())

    thickness = 4
    # print("imgpts[3]  ", imgpts[3])
    # print("imgpts[2]  ", imgpts[2])
    # print("imgpts[1]  ", imgpts[1])
    im = cv.line(im, frame_centre, tuple(imgpts[3].ravel()), (255,0,0), thickness, cv.LINE_AA)
    im = cv.line(im, frame_centre, tuple(imgpts[2].ravel()), (0,255,0), thickness, cv.LINE_AA)
    im = cv.line(im, frame_centre, tuple(imgpts[1].ravel()), (0,0,255), thickness, cv.LINE_AA)

    # cv.imshow("image", im)
    # cv.waitKey(0)

    return im


def get_pose(pts3d_mm, pts2d, cam_mat, dist, BOOL_SHOW_AXIS, AXIS_LENGTH, is_show=False):
    valid, rvec_pred, tvec_pred = cv.solvePnP(pts3d_mm, pts2d, cam_mat, dist)
    if valid and BOOL_SHOW_AXIS and is_show:
        # print(tvec_pred)
        # print("show image")
        show_axis(im, rvec_pred, tvec_pred, cam_mat, dist, AXIS_LENGTH)
    return valid, rvec_pred, tvec_pred#, inliers

class Keydot:
  def __init__(self,config):
    """ Load Object Tracking parameters """
    DIST_BETWEEN_BLOBS_MM = 0.595*1.5 # [mm]
    self.AXIS_LENGTH = 4
    self.PATTERN_SHAPE = (3, 7)
    self.BOOL_SHOW_AXIS = False

    self.pts3d_mm = get_3d_obj_coordinate(DIST_BETWEEN_BLOBS_MM, self.PATTERN_SHAPE)
    self.blob_detector = create_blob_detector()
    # self.cam_mat, self.dist = load_camera_parameters(CAM_CALIB)

    # self.cam_mat = np.array([ 812.2798, 0, 565.4246,
    #         0., 809.9507 , 287.3965,
    #         0., 0., 1. ]).reshape((3,3))
    # self.dist = np.array([0.3507, 0.5291,0,0,0])

  #   intrinsicMatrix =
  #     818.8713         0  527.3578
  #           0  824.3040  265.6600
  #           0         0    1.0000
  #   distortionCoefficients =
  #  -0.3618    0.4040         0         0         0

    self.cam_mat = np.array(config['cam']['camera_matrix']['data']).reshape((3,3))
    self.dist = np.array(config['cam']['dist_coeff']['data'])

  
  def detect(self,im, is_show=False):
      # show_blob_detector_result(im, blob_detector)####Jacopo. It was saying blob_detector not defined so I commented it like in the BACKUP file
      pts2d = get_2d_coordinates(im, self.PATTERN_SHAPE, self.blob_detector)
      if pts2d is not None:
          valid, rvec_pred, tvec_pred = get_pose(self.pts3d_mm, pts2d, self.cam_mat, self.dist, self.BOOL_SHOW_AXIS, self.AXIS_LENGTH)
          if valid:
              rmat_pred, _ = cv.Rodrigues(rvec_pred)
              transf = np.concatenate((rmat_pred, tvec_pred), axis = 1)
              # save_pose_np(img_dir_path+"homo_keydots/{}".format(idx), transf)
              # print(transf)
              if is_show:
                im = show_axis(im, rvec_pred, tvec_pred, self.cam_mat, self.dist, 5)
              return im, rvec_pred, tvec_pred
      return im, None, None

class CharucoDetect:
  def __init__(self,config):
    """ Load Object Tracking parameters """
    if config['dataset']['dictionary'] == 'DICT_5X5_250':
        self.dictionary = cv.aruco.getPredefinedDictionary(dict=cv.aruco.DICT_5X5_250)
    if config['dataset']['in_MM']:
        squaresY=config['dataset']['squaresY']
        squaresX=config['dataset']['squaresX']
        squareLength=config['dataset']['squareLength']
        markerLength=config['dataset']['markerLength']
        ### For Older version
        self.board = cv.aruco.CharucoBoard_create(squaresY=config['dataset']['squaresY'],
                                            squaresX=config['dataset']['squaresX'],
                                            squareLength=config['dataset']['squareLength'],
                                            markerLength=config['dataset']['markerLength'],
                                            dictionary=self.dictionary)

                                      
    self.camera_matrix = np.array(config['cam']['camera_matrix']['data']).reshape((3,3))
    self.dist_coefs = np.array(config['cam']['dist_coeff']['data'])

  
  def detect(self,im):
    # im = cv.imread('/home/hs3/catkin_ws/src/dvrk_record_video/test.png')
    im, rvec_pred, tvec_pred = Charuco_detect(im,self.dictionary, self.camera_matrix, self.dist_coefs, self.board)
    if im is not None and rvec_pred is not None:
        im = show_axis(im, rvec_pred, tvec_pred, self.camera_matrix, self.dist_coefs, 5)
        return im, rvec_pred, tvec_pred
    return im, None, None


class Sync:
  def __init__(self, path_package, config):
    """ Load settings from config file """
    self.is_visualization_on = config["show_visualization"]
    self.save_frames_timestamp = config["save_frames_timestamp"]
    self.discard_individual_frames = config["vid"]["discard_individual_frames"]
    self.fps = float(config["vid"]["fps"])
    self.is_otf_compression_on = config["on_the_fly_lossless_compression"]["is_on"]
    self.is_vid_on = False
    self.frame_pad = "07d"
    self.frame_frmt = ".bmp" # To keep it lossless and since `.bmp` is the fastest to write
    if self.is_otf_compression_on:
      self.set_up_otf_compression(config["on_the_fly_lossless_compression"]["format_opti"],
                                  config["on_the_fly_lossless_compression"]["n_cores"],
                                  config["on_the_fly_lossless_compression"]["counter_between_runs"])
    self.set_up_output_dir(path_package, config["output_dir"])
    self.set_up_output_file_paths(config["vid"]["format"])
    if self.is_vid_on:
      self.set_up_ffmpeg_command(config["vid"]["codec"], config["vid"]["crf"])
    """ Initialize flags for processing """
    self.is_shutdown = False
    self.is_processing = False
    self.is_new_msg = False
    self.counter_save = 0
    """ Subscribe to camera topics """
    self.data_sub_im1 = message_filters.Subscriber(config["rostopic"]["cam1"], Image)
    self.data_sub_im2 = message_filters.Subscriber(config["rostopic"]["cam2"], Image)
    self.ats = message_filters.ApproximateTimeSynchronizer([self.data_sub_im1,
                                                            self.data_sub_im2],
                                                            queue_size=500,
                                                            slop=1./self.fps)
    self.bridge = CvBridge()
    self.initialize_recording()
    time.sleep(1) # self.ats doesn't init if too quick (CPU issue?)
    self.ats.registerCallback(self.sync_callback)

    self.is_save=False


  def get_pose(self,pts3d_mm, pts2d, cam_mat, dist, BOOL_SHOW_AXIS, AXIS_LENGTH, is_show=False):
      valid, rvec_pred, tvec_pred = cv.solvePnP(pts3d_mm, pts2d, cam_mat, dist)
      if valid and BOOL_SHOW_AXIS and is_show:
          show_axis(im, rvec_pred, tvec_pred, cam_mat, dist, AXIS_LENGTH)
      return valid, rvec_pred, tvec_pred#, inliers

  def get_2d_coordinates(self,im, PATTERN_SHAPE, z):
      ret, corners = cv.findCirclesGrid(im, PATTERN_SHAPE, flags=(cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING), blobDetector=blob_detector)
      if ret:
          return corners
      return None


  def split_list(self, l, n):
    k, m = divmod(len(l), n)
    return list(l[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


  def set_up_otf_compression(self, frmt_opti, n_cores_requested, counter_between_runs):
    """ On-the-fly compression `frame_frmt` -> `frame_frmt_opti` """
    self.frame_frmt_opti = frmt_opti # Convert `.bmp` to `frmt_opti` in separate processes, to save memory
    """ I will split the frames between processes according to their name's ending, e.g. %0.bpm, %1.bpm """
    ends_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_cores = min(n_cores_requested, 10) # Max we need 10 cores, since there are 10 numbers in total
    n_cores_max = mp.cpu_count()
    self.n_cores = min(n_cores, n_cores_max) # We cannot request more cores than what the computer is equipped with
    self.counter_between_runs = counter_between_runs
    self.ends_splitted = self.split_list(ends_all, self.n_cores)
    self.counter_compress = 0
    self.frame_compress_completed = True
    self.pool = mp.Pool(self.n_cores)


  def set_up_output_dir(self, path_package, out_dir):
    if not os.path.isabs(out_dir):
      out_dir = os.path.join(path_package, out_dir)
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)
      rospy.loginfo("Directory {} created".format(out_dir))
    self.out_dir = out_dir


  def set_up_ffmpeg_command(self, codec, crf):
    """ This is the command that will be run at the end of the frame collection, to compress the frames into a video """
    frame_frmt = self.frame_frmt
    if self.is_otf_compression_on:
      frame_frmt = self.frame_frmt_opti
    self.ffmpeg_command = ["ffmpeg",
                           "-r", "{}".format(self.fps),                                   # Frames per second
                           "-i", "{}_%{}{}".format(self.out, self.frame_pad, frame_frmt), # Input images
                           "-c:v", "{}".format(codec),                                    # Video codec
                           "-b:v", "0",                                                   # Set bitrate to 0, to adjust quality from the "crf" value
                           "-crf", "{}".format(crf),                                      # Constant quality encoding (the lower the better the quality)
                           "-strict", "experimental",                                     # To allow experimental codecs as well
                           "-an",                                                         # No audio
                           "{}".format(self.out_vid)]                                     # Output video path


  def set_up_output_file_paths(self, out_vid_format):
    self.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    self.out = os.path.join(self.out_dir, "{}".format(self.timestr))
    if self.save_frames_timestamp:
        self.out_stamp = "{}_timestamps.txt".format(self.out)
    self.out_vid = "{}{}".format(self.out, out_vid_format)


  def initialize_recording(self):
    rospy.loginfo("Recording started!")
    if self.save_frames_timestamp:
      self.f = open(self.out_stamp, "a")


  def sync_callback(self, msg_im1, msg_im2):
    if self.is_shutdown:
      # The callback may still be called when compressing the video at the end
      return
    try:
      im1 = np.frombuffer(msg_im1.data, dtype=np.uint8).reshape(msg_im1.height, msg_im1.width, -1)
      # im1 = self.bridge.imgmsg_to_cv2(msg_im1, "bgr8")
      # im2 = self.bridge.imgmsg_to_cv2(msg_im2, "bgr8")
    except CvBridgeError as e:
      print(e)
      return
    # self.frame = np.hstack((im1, im2))
    self.frame = im1
    frame_timestamp = msg_im1.header.stamp
    if self.is_save:
      """ Save image """
      frame_path = "{}_{:{}}{}".format(self.out, self.counter_save, self.frame_pad, self.frame_frmt)
      cv.imwrite(frame_path, self.frame)
      self.counter_save += 1
      if self.save_frames_timestamp:
        """ Save image timestamp """
        self.f.write('{}\n'.format(frame_timestamp))
      """ On-the-fly frame compression """
      if self.is_otf_compression_on:
        if self.counter_save % self.counter_between_runs == 0:
          if self.frame_compress_completed:
            self.start_processes_to_compress_otf()


  def log_result(self, n_files):
    self.counter_compress += n_files
    if self.counter_compress == self.counter_compress_goal:
      self.frame_compress_completed = True


  def start_processes_to_compress_otf(self):
    self.frame_compress_completed = False
    self.counter_compress_goal = self.counter_save
    for ends in self.ends_splitted:
      frame_path = "{}*{}{}".format(self.out, ends, self.frame_frmt)
      self.pool.apply_async(compress_frames, args=(frame_path, self.frame_frmt, self.frame_frmt_opti), callback=self.log_result)


  def stop_recording_and_compress_video(self):
    self.is_shutdown = True
    rospy.loginfo("Stopped recording new frames!")
    if self.is_visualization_on:
      cv.destroyAllWindows()
    time.sleep(2)
    if self.save_frames_timestamp:
      if not self.f.closed:
        self.f.close() # Close file with timestamps
    frame_frmt = self.frame_frmt
    if self.is_otf_compression_on:
      rospy.loginfo("Please wait until all the individual frames are compressed...")
      while not self.frame_compress_completed:
        time.sleep(3)
      rospy.loginfo("Running the last compression of frames... It may take a while, please wait")
      self.start_processes_to_compress_otf()
      self.pool.close()
      self.pool.join()
      frame_frmt = self.frame_frmt_opti
    frame_path = "{}*{}".format(self.out, frame_frmt)
    path_im_list = glob.glob(frame_path)
    if path_im_list:
      rospy.loginfo("Compressing recorded images into video. Please wait...")
      if self.is_vid_on:
        self.process = sp.Popen(self.ffmpeg_command, stdin=sp.PIPE)
        self.process.wait() # Wait for sub-process to finish
        if self.discard_individual_frames:
          rospy.loginfo("Discarding individual frames...")
          for path_im in path_im_list:
            os.remove(path_im)#os.path.join(self.out_dir, path_im))
    rospy.loginfo("All done!")


def get_config_data(path_package):
  path_config = os.path.join(path_package, 'config.yaml')
  with open(path_config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        return config
    except yaml.YAMLError as e:
        print(e)


def compress_frames(frame_path, frame_frmt, frame_frmt_opti):
  signal.signal(signal.SIGINT, signal.SIG_IGN) # Make it ignore Ctrl + C, not to interfere with ROS Ctrl + C
  files = glob.glob(frame_path)
  if files:
    for file in files:
      frame = cv.imread(file)
      cv.imwrite(file.replace(frame_frmt, frame_frmt_opti), frame)
      os.remove(file)
  return len(files)

def vec_to_eul(rvec_pred):
  signal = 1
  rot_eul = np.array([0., 0., 0.])

  if rvec_pred is None:
    signal = 0

  elif rvec_pred is not None:
    rmat_pred, _ = cv.Rodrigues(rvec_pred)
    # print("rmat_pred  ", rmat_pred)
    
    rot_angle = R.from_matrix(rmat_pred.reshape(3,3))
    rot_eul = rot_angle.as_euler("xyz", degrees=True)
    # print("rot_eul  ", rot_eul)
    
  # print(type(rot_eul))
  return signal, rot_eul









def main(args):
  rospy.loginfo("Initializing node...")
  rospy.init_node("node_dvrk_record_video", anonymous=True)
  rospack = rospkg.RosPack()
  name_package = "dvrk_record_video"
  path_package = rospack.get_path(name_package)
  config = get_config_data(path_package)
  sync = Sync(path_package, config)
  rospy.on_shutdown(sync.stop_recording_and_compress_video)
  rospy.loginfo("Recording individual frames, and then compress when finished recording...")
  r = rospy.Rate(sync.fps)
  kd_detect = Keydot(config)
  ch_detect = CharucoDetect(config)
  if sync.is_visualization_on:
    cv.namedWindow(name_package, cv.WINDOW_KEEPRATIO)
  
  # file_name = 'pairs_kd_dvrk_test'
  file_name = 'pairs_kd_dvrk_v20'
  save_path = os.path.join('/home/hs3/Documents/Jacopo',file_name)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  
  is_reproj = True

  

  HOST = '172.22.143.63'
  PORT = 30000
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  # s.bind((HOST, PORT))
  # s.listen(10) # socket queue of 10
  # conn, addr = s.accept()

  server_address = (HOST,PORT)
  s.connect(server_address)





  frame_shot_index = 0
  reprojection = True


  # basePSM_T_cam =\
  # np.array([[-8.68873492e-01, -2.65180721e-01,  4.18016794e-01,
  #         -7.89811610e+01],
  #     [-4.94017845e-01,  4.10401731e-01, -7.66496437e-01,
  #         1.58176773e+02],
  #     [ 3.17052617e-02, -8.72496192e-01, -4.87591194e-01,
  #         -6.35438284e+01],
  #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  #         1.00000000e+00]])
  # cam_T_basePSM =\
  # np.array([[-8.68873492e-01, -4.94017845e-01,  3.17052617e-02,
  #         1.15321853e+01],
  #     [-2.65180721e-01,  4.10401731e-01, -8.72496192e-01,
  #         -1.41302051e+02],
  #     [ 4.18016794e-01, -7.66496437e-01, -4.87591194e-01,
  #         1.23273974e+02],
  #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  #         1.00000000e+00]])

  # X =\
  # np.array([[-0.1281,  0.031 , -0.9913, -1.8891],
  #      [-0.9916,  0.0129,  0.1285, -0.2209],
  #      [ 0.0168,  0.9994,  0.0291,  8.8926],
  #      [ 0.    ,  0.    ,  0.    ,  1.    ]])
  # inv_X=\
  # np.array([[-0.1281, -0.9916,  0.0168, -0.61  ],
  #       [ 0.031 ,  0.0129,  0.9994, -8.8262],
  #       [-0.9913,  0.1285,  0.0291, -2.1026],
  #       [ 0.    ,  0.    ,  0.    ,  1.    ]])
  # Y =\
  # np.array([[ -0.9758,  -0.037 ,   0.2156, -68.1321],
  #       [ -0.2054,   0.4937,  -0.845 , 164.7486],
  #       [ -0.0751,  -0.8688,  -0.4894, -27.4636],
  #       [  0.    ,   0.    ,   0.    ,   1.    ]])
  # inv_Y =\
  # np.array([[  -0.9758,   -0.2054,   -0.0751,  -34.7017],
  #       [  -0.037 ,    0.4937,   -0.8688, -107.7242],
  #       [   0.2156,   -0.845 ,   -0.4894,  140.4622],
  #       [   0.    ,    0.    ,    0.    ,    1.    ]])





  while not rospy.is_shutdown():
    """ Show what is being recorded """
    if sync.is_visualization_on:
      if sync.frame is not None:
        resized = resize_img(sync.frame)
        # print("resized ---    ", resized)

        resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

        

        
        resized, kd_rvec_pred, kd_tvec_pred = kd_detect.detect(resized,True) ### detect pose of keydot pattern
        # print("kd_tvec_pred 1 :", kd_tvec_pred)
        # kd_tvec_pred = kd_tvec_pred*0.001
        # print("kd_tvec_pred 2 :", kd_tvec_pred)

        # if kd_rvec_pred is not None:
        if True:
          # # print("resized, kd_rvec_pred, kd_tvec_pred = ", resized, "  xxx  ", kd_rvec_pred,"  xxx  ", kd_tvec_pred)
          # kd_rmat_pred, _ = cv.Rodrigues(kd_rvec_pred)
          # print("rot:   ", kd_rmat_pred)
          # print("trans: ", kd_tvec_pred)
          # # # print(np.size(kd_tvec_pred))


          # ##Record for AX=YB
          # # kd_rmat_pred,_ = cv.Rodrigues(kd_rvec_pred)
          # # kd_pose  = np.concatenate((kd_rmat_pred,kd_tvec_pred).axis=0)
          # # # np.save('{}.npy'.format(frame_idx),kd_pose)
          # # savemat("A/c{}.mat".format(frame_index), kd_pose)
          # if True:
          

          

          







          resized, ch_rvec_pred, ch_tvec_pred = ch_detect.detect(resized) ### detect pose of charuco
          # # print("resized, ch_rvec_pred, ch_tvec_pred = ", resized, ch_rvec_pred, ch_tvec_pred)
          # print("-------------------")
          # print("-------------------")
          # print("-------------------")
          # print("-------------------")
          # print("-------------------")



          

          kd_ch_signal = np.array([0.0])

          signal_kd, eul_angles_kd = vec_to_eul(kd_rvec_pred)
          signal_ch, eul_angles_ch = vec_to_eul(ch_rvec_pred)

          # print("eul_angles: ",no_signal, "   ", eul_angles)
          # print(type(eul_angles))

          # Convert into vectors first
          ch_vec = np.concatenate((ch_rvec_pred,ch_tvec_pred), axis=None)
          kd_vec = np.concatenate((kd_rvec_pred,kd_tvec_pred), axis=None)

          # Check for the signal status and update vectors to 0 vectors depending on the signal status
          if signal_kd == 1 and signal_ch == 1:
            # Both signals are 1
            kd_ch_signal = np.array([3.0])

          elif signal_kd == 0 and signal_ch == 1:
            # Only kd is zero
            kd_ch_signal = np.array([2.0])
            kd_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
          elif signal_kd == 1 and signal_ch == 0:
            # Only ch is zero
            kd_ch_signal = np.array([1.0])
            ch_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
          elif signal_kd == 0 and signal_ch == 0:
            # Both ch and kd is zero
            kd_ch_signal = np.array([0.0])
            ch_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            kd_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
          
            

          # # Print to check
          # print("kd_vec: ", kd_vec)
          # print("ch_vec: ", ch_vec)
          # print('kd_ch_signal: ', kd_ch_signal)



          # Concatenate channel signal status and vectors
          kd_ch_vec = np.concatenate((kd_ch_signal,kd_vec,ch_vec), axis=None)
          message = ','.join(str(round(x,8)) for x in kd_ch_vec)

          # Send Message
          s.sendall(message.encode())


          # print("1 ")
          if True:  ###133 To record pairs for hand-eye calibration ( to put True on node_dvrK_track.py as well if recording)
            ##Receiving the dVRK arm transformation matrix from server
            dataFromServer = s.recv(1024)
            # print("dataFromServer",dataFromServer.decode())
            LCread_arm_str = dataFromServer.decode()
            LCread_arm = LCread_arm_str.split(",")

            dvrk_Rot = np.eye(3)
            dvrk_Trans = np.zeros((1,3))
            # print("2 ")
            # print("dvrk_Trans ", dvrk_Trans)
            if len(LCread_arm) == 12: #12 = 3x3 rotation amtrix and 1x3 translation
              dvrk_Rot[0][0] = float(LCread_arm[0])
              dvrk_Rot[0][1] = float(LCread_arm[1])
              dvrk_Rot[0][2] = float(LCread_arm[2])#
              dvrk_Rot[1][0] = float(LCread_arm[3])
              dvrk_Rot[1][1] = float(LCread_arm[4])
              dvrk_Rot[1][2] = float(LCread_arm[5])#
              dvrk_Rot[2][0] = float(LCread_arm[6])
              dvrk_Rot[2][1] = float(LCread_arm[7])
              dvrk_Rot[2][2] = float(LCread_arm[8])#

              dvrk_Trans[0][0] = float(LCread_arm[9])
              dvrk_Trans[0][1] = float(LCread_arm[10])
              dvrk_Trans[0][2] = float(LCread_arm[11])

              dvrk_arm_pose = np.concatenate((dvrk_Rot,dvrk_Trans.T), axis=1)
              # print("dvrk_arm_pose ", dvrk_arm_pose)
            




              # ##### Check that projection of hand-eye calibration frame is correct
              # if reprojection:
              #   proj_cam_pose = np.dot(cam_T_basePSM,dvrk_arm_pose)
              #   proj_rmat, proj_trans = unpack_homo(proj_cam_pose)
              #   show_axis(resized, proj_rmat, proj_trans , cam_mat, dist, AXIS_LENGTH)
              # ######




            pressed_key = cv.waitKey(1)
            # print("pressed_key  ", pressed_key)

            if kd_rvec_pred is not None:
              kd_rmat_pred,_ = cv.Rodrigues(kd_rvec_pred)
              kd_pose  = np.concatenate((kd_rmat_pred,kd_tvec_pred), axis=1)
              # # np.save('{}.npy'.format(frame_idx),kd_pose)
              # savemat("A/c{}.mat".format(frame_index), kd_pose)

              # print("kd_pose ", kd_pose)  ##type(kd_pose) is np.array  

            # print(" len(LCread_arm) ", len(LCread_arm))
            # print(" (LCread_arm) ", LCread_arm)
            # if len(LCread_arm) == 12 and kd_rvec_pred is not None:
            if len(LCread_arm) == 12:
              if pressed_key == 13: #13 = enter
                print('save!')
                frame_shot_index += 1
                # savemat('kd{}.mat'.format(frame_shot_index), kd_pose)            #NOTE: if you press enter before the kd_pose is defined earlier, then you'll NOT get an error because this " if len(LCread_arm) == 12 and kd_rvec_pred is not None:" is not satisfied
                # savemat('/home/hs3/catkin_ws/src/dvrk_record_video/dvrk_arm_pose/dvrk_arm{}.mat'.format(frame_shot_index), dvrk_arm_pose)#.. same thing for dvrk_arm_pose
                np.save(save_path+'/kd_{}.npy'.format(frame_shot_index),kd_pose)
                np.save(save_path+'/dvrk_arm_{}.npy'.format(frame_shot_index),dvrk_arm_pose)
                print("kd_pose   ", kd_pose)
                print("dvrk_arm_pose   ", dvrk_arm_pose)
                print('----')
                print('----')
                print('----')
                print('----')
                print('----')

          if is_reproj:
            dataFromServer = s.recv(1024)
            # print("dataFromServer",dataFromServer.decode())
            LCread_arm_str = dataFromServer.decode()
            LCread_arm = LCread_arm_str.split(",")

            dvrk_Rot = np.eye(3)
            dvrk_Trans = np.zeros((1,3))
            # print("2 ")
            # print("dvrk_Trans ", dvrk_Trans)
            if len(LCread_arm) == 12: #12 = 3x3 rotation amtrix and 1x3 translation
              # print("receiving dvrk position, now projecting")
              dvrk_Rot[0][0] = float(LCread_arm[0])
              dvrk_Rot[0][1] = float(LCread_arm[1])
              dvrk_Rot[0][2] = float(LCread_arm[2])#
              dvrk_Rot[1][0] = float(LCread_arm[3])
              dvrk_Rot[1][1] = float(LCread_arm[4])
              dvrk_Rot[1][2] = float(LCread_arm[5])#
              dvrk_Rot[2][0] = float(LCread_arm[6])
              dvrk_Rot[2][1] = float(LCread_arm[7])
              dvrk_Rot[2][2] = float(LCread_arm[8])#

              dvrk_Trans[0][0] = float(LCread_arm[9])
              dvrk_Trans[0][1] = float(LCread_arm[10])
              dvrk_Trans[0][2] = float(LCread_arm[11])

              dvrk_arm_pose = np.concatenate((dvrk_Rot,dvrk_Trans.T), axis=1)
              dvrk_arm_pose = np.concatenate((dvrk_arm_pose,np.array([[0,0,0,1]])), axis=0)
              # print("dvrk_arm_pose ", dvrk_arm_pose)





              # X =\
              # np.array([[-0.11098287,  0.05165553, -0.99247897, -3.22788445],
              #     [-0.99166621,  0.05999053,  0.11401431,  0.43888095],
              #     [ 0.06542881,  0.9968615 ,  0.04456712, 12.76455803],
              #     [ 0.        ,  0.        ,  0.        ,  1.        ]])
              # cam_T_basePSM =\
              # np.array([[-8.68873492e-01, -4.94017845e-01,  3.17052617e-02,
              #         1.19636518e+01],
              #     [-2.65180721e-01,  4.10401731e-01, -8.72496192e-01,
              #         -1.43112841e+02],
              #     [ 4.18016794e-01, -7.66496437e-01, -4.87591194e-01,
              #         1.24668556e+02],
              #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              #         1.00000000e+00]])


              


              # X =\
              # np.array([[-0.1659,  0.0784, -0.983 , -0.7506],
              #   [-0.9751,  0.1355,  0.1754,  0.7252],
              #   [ 0.147 ,  0.9877,  0.054 , 11.9906],
              #   [ 0.    ,  0.    ,  0.    ,  1.    ]])
              # inv_X=\
              # np.array([[ -0.1659,  -0.9751,   0.147 ,  -1.1795],
              #     [  0.0784,   0.1355,   0.9877, -11.8822],
              #     [ -0.983 ,   0.1754,   0.054 ,  -1.5121],
              #     [  0.    ,   0.    ,   0.    ,   1.    ]])
              # basePSM_T_cam =\
              # np.array([[ -0.9708,  -0.1468,   0.1898, -50.1703],
              #     [ -0.2384,   0.5006,  -0.8322, 158.0005],
              #     [  0.0272,  -0.8532,  -0.521 , -42.3297],
              #     [  0.    ,   0.    ,   0.    ,   1.    ]])
              # cam_T_basePSM =\
              # np.array([[  -0.9708,   -0.2384,    0.0272,   -9.8834],
              #     [  -0.1468,    0.5006,   -0.8532, -122.5706],
              #     [   0.1898,   -0.8322,   -0.521 ,  118.9612],
              #     [   0.    ,    0.    ,    0.    ,    1.    ]])
              X =\
              np.array([[-0.1281,  0.031 , -0.9913, -1.8891],
                    [-0.9916,  0.0129,  0.1285, -0.2209],
                    [ 0.0168,  0.9994,  0.0291,  8.8926],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[-0.1281, -0.9916,  0.0168, -0.61  ],
                    [ 0.031 ,  0.0129,  0.9994, -8.8262],
                    [-0.9913,  0.1285,  0.0291, -2.1026],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              ### Y
              Y =\
              np.array([[ -0.9758,  -0.037 ,   0.2156, -68.1321],
                    [ -0.2054,   0.4937,  -0.845 , 164.7486],
                    [ -0.0751,  -0.8688,  -0.4894, -27.4636],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              ### inv Y
              inv_Y =\
              np.array([[  -0.9758,   -0.2054,   -0.0751,  -34.7017],
                    [  -0.037 ,    0.4937,   -0.8688, -107.7242],
                    [   0.2156,   -0.845 ,   -0.4894,  140.4622],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])

              ###wrong, I used v6 calibration instead of 7
              X =\
              np.array([[-0.1281,  0.031 , -0.9913, -1.8891],
                  [-0.9916,  0.0129,  0.1285, -0.2209],
                  [ 0.0168,  0.9994,  0.0291,  8.8926],
                  [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[-0.1281, -0.9916,  0.0168, -0.61  ],
                  [ 0.031 ,  0.0129,  0.9994, -8.8262],
                  [-0.9913,  0.1285,  0.0291, -2.1026],
                  [ 0.    ,  0.    ,  0.    ,  1.    ]])
              Y =\
              np.array([[ -0.9758,  -0.037 ,   0.2156, -68.1321],
                  [ -0.2054,   0.4937,  -0.845 , 164.7486],
                  [ -0.0751,  -0.8688,  -0.4894, -27.4636],
                  [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_Y =\
              np.array([[  -0.9758,   -0.2054,   -0.0751,  -34.7017],
                  [  -0.037 ,    0.4937,   -0.8688, -107.7242],
                  [   0.2156,   -0.845 ,   -0.4894,  140.4622],
                  [   0.    ,    0.    ,    0.    ,    1.    ]])

              


              X =\
              np.array([[ 0.0652,  0.0212, -0.9976, -3.2298],
                    [ 0.1317,  0.9908,  0.0297, -5.9645],
                    [ 0.9891, -0.1334,  0.0618, 17.1157],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[  0.0652,   0.1317,   0.9891, -15.9332],
                    [  0.0212,   0.9908,  -0.1334,   8.2614],
                    [ -0.9976,   0.0297,   0.0618,  -4.1035],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              Y =\
              np.array([[ -0.2295,  -0.5786,   0.7826, -84.4034],
                    [ -0.9719,   0.0925,  -0.2166, -12.8707],
                    [  0.053 ,  -0.8103,  -0.5836, -72.9198],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_Y =\
              np.array([[  -0.2295,   -0.9719,    0.053 ,  -28.0158],
                    [  -0.5786,    0.0925,   -0.8103, -106.738 ],
                    [   0.7826,   -0.2166,   -0.5836,   20.7126],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])



              X =\
              np.array([[ 0.0568, -0.9972,  0.0482,  3.2369],
       [ 0.1531, -0.039 , -0.9874, -8.0395],
       [ 0.9866,  0.0635,  0.1504,  7.1108],
       [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[ 0.0568,  0.1531,  0.9866, -5.9686],
                    [-0.9972, -0.039 ,  0.0635,  2.4628],
                    [ 0.0482, -0.9874,  0.1504, -9.1643],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              Y =\
              np.array([[  0.328 ,  -0.4195,   0.8464, -90.9284],
                    [ -0.941 ,  -0.2243,   0.2535, -51.0612],
                    [  0.0835,  -0.8796,  -0.4683, -35.4656],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_Y =\
              np.array([[  0.328 ,  -0.941 ,   0.0835, -15.2618],
                    [ -0.4195,  -0.2243,  -0.8796, -80.7928],
                    [  0.8464,   0.2535,  -0.4683,  73.2982],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])


              X =\
              np.array([[ 0.0742, -0.9966,  0.0344,  3.1458],
                    [ 0.3807, -0.0035, -0.9247, -7.9434],
                    [ 0.9217,  0.0817,  0.3792,  8.3318],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[  0.0742,   0.3807,   0.9217,  -4.8891],
                    [ -0.9966,  -0.0035,   0.0817,   2.4263],
                    [  0.0344,  -0.9247,   0.3792, -10.6124],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              Y =\
              np.array([[  0.3425,  -0.4426,   0.8287, -93.5468],
                    [ -0.9348,  -0.2487,   0.2535, -47.6214],
                    [  0.0939,  -0.8615,  -0.4989, -36.0342],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_Y =\
              np.array([[  0.3425,  -0.9348,   0.0939,  -9.0982],
                    [ -0.4426,  -0.2487,  -0.8615, -84.2924],
                    [  0.8287,   0.2535,  -0.4989,  71.619 ],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])

              
              X =\
              np.array([[ 0.0024, -0.9965,  0.0836,  1.7059],
              [ 0.09  , -0.0831, -0.9925, -8.0014],
              [ 0.9959,  0.0099,  0.0895,  3.9685],
              [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[ 0.0024,  0.09  ,  0.9959, -3.2361],
                    [-0.9965, -0.0831,  0.0099,  0.9958],
                    [ 0.0836, -0.9925,  0.0895, -8.4391],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              Y =\
              np.array([[   0.1648,   -0.5991,    0.7835,  -60.1267],
                    [  -0.9793,   -0.0045,    0.2025, -102.04  ],
                    [  -0.1178,   -0.8006,   -0.5874,  -30.2158],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])
              inv_Y =\
              np.array([[  0.1648,  -0.9793,  -0.1178, -93.5773],
                    [ -0.5991,  -0.0045,  -0.8006, -60.6735],
                    [  0.7835,   0.2025,  -0.5874,  50.024 ],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])

              

              X =\
              np.array([[ 0.0843, -0.9963,  0.0171,  1.1467],
                    [ 0.073 , -0.0109, -0.9973, -9.3246],
                    [ 0.9938,  0.0853,  0.0718,  5.2887],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[ 0.0843,  0.073 ,  0.9938, -4.6715],
                    [-0.9963, -0.0109,  0.0853,  0.5893],
                    [ 0.0171, -0.9973,  0.0718, -9.6986],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              Y =\
              np.array([[   0.1617,   -0.598 ,    0.7851,  -60.0049],
                    [  -0.9812,   -0.0121,    0.1928, -105.8451],
                    [  -0.1058,   -0.8014,   -0.5886,  -31.0282],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])
              inv_Y =\
              np.array([[  0.1617,  -0.9812,  -0.1058, -97.4335],
                    [ -0.598 ,  -0.0121,  -0.8014, -62.0287],
                    [  0.7851,   0.1928,  -0.5886,  49.251 ],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])


              
              X =\
              np.array([[  0.0812,  -0.9963,  -0.0272,   1.5622],
                    [  0.0512,   0.0314,  -0.9982, -10.3433],
                    [  0.9954,   0.0796,   0.0536,   3.2134],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_X=\
              np.array([[  0.0812,   0.0512,   0.9954,  -2.7958],
                    [ -0.9963,   0.0314,   0.0796,   1.6253],
                    [ -0.0272,  -0.9982,   0.0536, -10.4543],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              Y =\
              np.array([[   0.1599,   -0.6065,    0.7789,  -60.0621],
                    [  -0.9808,   -0.0084,    0.1948, -105.2985],
                    [  -0.1116,   -0.7951,   -0.5962,  -30.0852],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])
              inv_Y =\
              np.array([[  0.1599,  -0.9808,  -0.1116, -97.032 ],
                    [ -0.6065,  -0.0084,  -0.7951, -61.2298],
                    [  0.7789,   0.1948,  -0.5962,  49.3562],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])


              X =\
              np.array([[ 0.032 , -0.9994,  0.0147, -0.6469],
                    [ 0.0743, -0.0123, -0.9972, -8.3045],
                    [ 0.9967,  0.033 ,  0.0739,  4.7762],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              inv_X=\
              np.array([[ 0.032 ,  0.0743,  0.9967, -4.1227],
                    [-0.9994, -0.0123,  0.033 , -0.9061],
                    [ 0.0147, -0.9972,  0.0739, -8.6242],
                    [ 0.    ,  0.    ,  0.    ,  1.    ]])
              Y =\
              np.array([[   0.1528,   -0.5964,    0.788 ,  -60.0653],
                    [  -0.9832,   -0.0113,    0.1821, -104.0522],
                    [  -0.0997,   -0.8026,   -0.5881,  -30.5349],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])
              inv_Y =\
              np.array([[  0.1528,  -0.9832,  -0.0997, -96.1741],
                    [ -0.5964,  -0.0113,  -0.8026, -61.5029],
                    [  0.788 ,   0.1821,  -0.5881,  48.322 ],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])


              

              X =\
              np.array([[  0.0914,  -0.9958,  -0.0091,   2.6136],
                    [  0.162 ,   0.0239,  -0.9865, -10.491 ],
                    [  0.9825,   0.0887,   0.1635,   5.2343],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_X=\
              np.array([[  0.0914,   0.162 ,   0.9825,  -3.6821],
                    [ -0.9958,   0.0239,   0.0887,   2.3886],
                    [ -0.0091,  -0.9865,   0.1635, -11.1815],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              Y =\
              np.array([[   0.1677,   -0.5974,    0.7842,  -83.2143],
                    [  -0.9843,   -0.0566,    0.1673, -100.3075],
                    [  -0.0555,   -0.7999,   -0.5975,  -41.2848],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])
              inv_Y =\
              np.array([[  0.1677,  -0.9843,  -0.0555, -87.0711],
                    [ -0.5974,  -0.0566,  -0.7999, -88.4189],
                    [  0.7842,   0.1673,  -0.5975,  57.3704],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])


              X =\
              np.array([[  0.1169,  -0.9928,  -0.0255,   2.2893],
                    [  0.2751,   0.057 ,  -0.9597, -10.2717],
                    [  0.9543,   0.1052,   0.2798,   6.1961],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_X=\
              np.array([[  0.1169,   0.2751,   0.9543,  -3.3547],
                    [ -0.9928,   0.057 ,   0.1052,   2.2069],
                    [ -0.0255,  -0.9597,   0.2798, -11.5332],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              Y =\
              np.array([[   0.2499,   -0.663 ,    0.7057,  -76.1806],
                    [  -0.962 ,   -0.0874,    0.2585, -109.9155],
                    [  -0.1097,   -0.7435,   -0.6596,  -35.8606],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])
              inv_Y =\
              np.array([[  0.2499,  -0.962 ,  -0.1097, -90.6453],
                    [ -0.663 ,  -0.0874,  -0.7435, -86.7722],
                    [  0.7057,   0.2585,  -0.6596,  58.5224],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])



              X =\
              np.array([[  0.0499,  -0.9944,   0.093 ,  -0.3836],
                    [  0.2827,  -0.0752,  -0.9562, -11.8402],
                    [  0.9579,   0.0741,   0.2774,   3.5277],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              inv_X=\
              np.array([[  0.0499,   0.2827,   0.9579,  -0.0124],
                    [ -0.9944,  -0.0752,   0.0741,  -1.5333],
                    [  0.093 ,  -0.9562,   0.2774, -12.265 ],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])
              Y =\
              np.array([[   0.2558,   -0.6355,    0.7285,  -76.9687],
                    [  -0.9624,   -0.0957,    0.2544, -107.0652],
                    [  -0.0919,   -0.7662,   -0.636 ,  -34.3345],
                    [   0.    ,    0.    ,    0.    ,    1.    ]])
              inv_Y =\
              np.array([[  0.2558,  -0.9624,  -0.0919, -86.5018],
                    [ -0.6355,  -0.0957,  -0.7662, -85.4668],
                    [  0.7285,   0.2544,  -0.636 ,  61.4723],
                    [  0.    ,   0.    ,   0.    ,   1.    ]])















              # kd_reprojkd_offset = np.array([[ -0.90662495,   0.32587679,  -0.26805414,  51.96291493],
              # [  0.20950649,   0.89889278,   0.38462406, -30.49319149],
              # [  0.36627789,   0.29258422,  -0.88331808,  90.91238993],
              # [  0.,           0.,           0.,           1.        ]])
  #             kd_reprojkd_offset = np.array([[ 9.87405364e-01, -1.58218320e-01, -2.87034189e-04, -1.51419211e+00],
  #  [ 1.58033433e-01,  9.85652721e-01,  5.77684728e-02, -5.85027927e+00],
  #  [-8.79452763e-03, -5.71411015e-02,  9.98349760e-01, -3.68352413e+00],
  #  [ 0.00000000e+0,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
              is_option1 = True
              if is_option1:
                ### option1
                kd_reprojkd_offset = np.array([[ 0.98415125,  0.16670929,  0.06024322,  2.23766316],
    [-0.16549588,  0.98578593, -0.02646875 , 2.65879879],
    [-0.06376235,  0.01606048,  0.99785185 , 3.04691505],
    [ 0. ,         0.  ,        0.   ,       1.        ]])
      #           kd_reprojkd_offset = np.array([[-0.94722132,  0.18142098, -0.26421271, 25.02230603],
      # [ 0.26451761,  0.90791075, -0.32499719, 52.19683491],
      # [ 0.18087494, -0.37782007, -0.90802848, 99.37549204],
      # [ 0.,          0.,          0.,          1.        ]])
          #       kd_reprojkd_offset = np.array([[-0.947174,    0.18142225, -0.26438146, 25.04805105],
          # [ 0.26458812,  0.90788665, -0.32500714, 52.19817859],
          # [ 0.18101957, -0.37787738, -0.9079758,  99.36704751],
          # [ 0.,          0.,          0.,          1.        ]])
                kd_reprojkd_offset = np.array([[ 0.99921803,  0.03780851,  0.01150773,  1.4671176 ],
  [-0.03762697,  0.99932144, -0.00813041, -0.62271875],
  [-0.01181085,  0.00766081,  0.99989772, -1.21920134],
  [ 0.,          0.,          0.,          1.        ]])
                kd_reprojkd_offset = np.array([[ 0.99941374, -0.0156126,   0.03170734, -1.97581706],
  [ 0.01643807,  0.9995817,  -0.02711513, -0.50762614],
  [-0.03128088,  0.02758183,  0.99906569,  4.63111823],
  [ 0.,          0.,          0.,          1.        ]])
                kd_reprojkd_offset  = np.array([[ 9.87449059e-01,  1.55928587e-01, -2.56877869e-02,  1.37917288e+00],
  [-1.55946865e-01,  9.87846081e-01,  1.22431886e-03, -1.01045229e+00],
  [ 2.55872001e-02,  2.77340901e-03,  9.99632448e-01,  7.96930367e-01],
  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

                kd_reprojkd_offset = np.array([[ 2.95723966e-01, -3.53475452e-01,  8.87503741e-01,  1.19800340e+02],
                [ 9.53886400e-01,  1.60505627e-01, -2.53872955e-01, -8.80583879e+01],
                [-5.27181063e-02,  9.21596751e-01,  3.84479456e-01, -3.90595330e+01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

                kd_reprojkd_offset = np.array([[ 1.14612773e-01, -3.76635868e-01,  9.19274389e-01,  1.46910801e+01],
                [ 9.40293456e-01,  3.39878454e-01,  2.20331587e-02, -4.25352106],
                [-3.20695517e-01,  8.61780897e-01,  3.92931110e-01, -1.71570511],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

                kd_reprojkd_offset = np.array([[  0.47939469,  -0.13174879,   0.86769666, 101.86846351],
                [ -0.23096921,   0.93496672,   0.26955251, -11.14504145],
                [ -0.84667311,  -0.3295352,    0.41764348,  25.54618066],
                [  0.,           0.,           0.,           1.        ]])

                # kd_reprojkd_offset = np.array([[  0.71845391,   0.32111064,   0.61707897, 162.32509152],
                # [ -0.55645854,   0.79770605,   0.23269398,  73.85506706],
                # [ -0.41747292,  -0.51058812,   0.75161927,  51.30212024],
                # [  0.,           0.,           0.,           1.        ]])

                kd_reprojkd_offset = np.array([[ 0.26894229,  0.96225262, -0.04320716,  0.49454298],
                [-0.96282113,  0.26985432,  0.01320572, 11.62899069],
                [ 0.02441398,  0.03809048,  0.99904857,  3.22711622],
                [ 0.,          0.,          0.,          1.        ]])

                kd_reprojkd_offset = np.array([[ 0.99139111,  0.02133653,  0.12862053,  0.6324936 ],
                [-0.02322853,  0.99958384,  0.01391291,  0.3331754 ],
                [-0.12818767, -0.0167787,   0.99156807, -0.52443155],
                [ 0.,          0.,          0.,          1.        ]])


                kd_reprojkd_offset = np.array([[ 0.99998402, -0.00300231,  0.00125179,  0.10372551],
                [ 0.00301436,  0.99990756,  0.01772875,  0.37169741],
                [-0.00135721, -0.01779428,  0.99981676,  0.43668062],
                [ 0.,          0.,          0.,          1.        ]])












              else:
              ### option2
                kd_reprojkd_offset = np.array([[ 9.84209280e-01, -1.77014446e-01, -4.89584605e-03, -2.77225903e+00],
    [ 1.77108788e-01,  9.84021598e-01 , 1.26796955e-02, -1.55114245e+00],
    [ 2.63432417e-03, -1.34218269e-02,  9.99915164e-01, -1.89622424e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                kd_reprojkd_offset = np.array([[ 0.98415125,  0.16670929,  0.06024322,  2.23766316],
                  [-0.16549588,  0.98578593, -0.02646875,  2.65879879],
                  [-0.06376235,  0.01606048,  0.99785185,  3.04691505],
                  [ 0.,          0.,          0.,          1.        ]])
                kd_reprojkd_offset = np.array([[ 0.99921803,  0.03780851,  0.01150773,  1.4671176 ],
    [-0.03762697,  0.99932144, -0.00813041, -0.62271875],
    [-0.01181085,  0.00766081,  0.99989772, -1.21920134],
    [ 0.,          0.,          0.,          1.        ]])
                kd_reprojkd_offset = np.array([[ 0.99904726,  0.03736124,  0.02405553,  1.31672616],
              [-0.0374599,   0.99934108,  0.00675554, -2.76741636],
              [-0.02372403, -0.00771589,  0.99964677, -2.08405972],
              [ 0.,          0.,          0.,          1.        ]])








              # cam_mat = np.array([ 812.2798, 0, 565.4246,
              #             0., 809.9507 , 287.3965,
              #             0., 0., 1. ]).reshape((3,3))

              # dist = np.array([ -0.3600, 0.5683, 0, 0, 0.])

              cam_mat = np.array([ 822.4536, 0, 520.9665,
                            0, 827.0876, 262.6696,
                            0., 0., 1.]).reshape((3,3))

              dist = np.array([ -0.3755, 0.4431, 0, 0, 0])
              AXIS_LENGTH = 10

              # if kd_rvec_pred is not None:
              #   kd_rmat_pred,_ = cv.Rodrigues(kd_rvec_pred)
              #   kd_pose  = np.concatenate((kd_rmat_pred,kd_tvec_pred), axis=1)
              #   kd_pose = np.concatenate((kd_pose,np.array([[0,0,0,1]])), axis=0)

              #1
              reproj_kd_pose = np.dot(inv_Y,dvrk_arm_pose)
              # print("reproj_kd_pose  1 ", reproj_kd_pose)
              reproj_kd_pose = np.dot(reproj_kd_pose,X)
              # print("reproj_kd_pose  1 1: ", reproj_kd_pose)
              #2
              # reproj_kd_pose = np.dot(Y,dvrk_arm_pose)
              # # print("reproj_kd_pose  1 ", reproj_kd_pose)
              # reproj_kd_pose = np.dot(reproj_kd_pose,inv_X)

              ###### if reproj_kd_pose[0,0]!=0.0:
              ######   print('reproj_kd_pose',reproj_kd_pose)


              # print("reproj_kd_pose  2 ", reproj_kd_pose)
              reproj_rvec_pred, reproj_tvec_pred = unpack_homo(reproj_kd_pose)

              ### reprojected pose BEFORE offset
              show_axis(resized, reproj_rvec_pred, reproj_tvec_pred, cam_mat, dist, 25)  #AXIS_LENGTH

              # kd_pose = np.concatenate((kd_pose,np.array([[0,0,0,1]])), axis=0)





              # ###########AAAAAAAAAA33 During calibration/compensation offset
              # is_option1 = True

              # if kd_rvec_pred is not None:
              #   # resized, kd_rvec_pred, kd_tvec_pred = kd_detect.detect(resized,False)
              #   # print("kd_tvec_pred -> ", kd_tvec_pred)
              #   kd_rmat_pred,_ = cv.Rodrigues(kd_rvec_pred)
              #   kd_pose  = np.concatenate((kd_rmat_pred,kd_tvec_pred), axis=1)
              #   kd_pose = np.concatenate((kd_pose,np.array([[0,0,0,1]])), axis=0)
              #   print("reproj_kd_pose   ::", reproj_kd_pose)
              #   print("kd_pose   ::", kd_pose)
              #   print("...")
              #   print("...")
              #   print("...")

              #   if is_option1:
              #     # option1
              #     kd_reprojkd_offset = np.dot(inv(reproj_kd_pose),kd_pose)
              #     reproj_kd_pose2 = np.dot(reproj_kd_pose,kd_reprojkd_offset)
              #   else:
              #     #option2
              #     # kd_reprojkd_offset = np.dot(kd_pose,inv(reproj_kd_pose))
              #     reproj_kd_pose2 = np.dot(kd_reprojkd_offset,reproj_kd_pose)


              #   print('kd_reprojkd_offset AAAAAyyyxz ',kd_reprojkd_offset)
              # ##########BBBBBBB



              ############################# BBBB33
              is_option1 = True

              # if kd_rvec_pred is not None:
              #   kd_rmat_pred,_ = cv.Rodrigues(kd_rvec_pred)
              #   kd_pose  = np.concatenate((kd_rmat_pred,kd_tvec_pred), axis=1)
              #   kd_pose = np.concatenate((kd_pose,np.array([[0,0,0,1]])), axis=0)

              if is_option1:
                # option1
                # kd_reprojkd_offset = np.dot(inv(reproj_kd_pose),kd_pose)
                reproj_kd_pose2 = np.dot(reproj_kd_pose,kd_reprojkd_offset)
              else:
                #option2
                # kd_reprojkd_offset = np.dot(kd_pose,inv(reproj_kd_pose))
                reproj_kd_pose2 = np.dot(kd_reprojkd_offset,reproj_kd_pose)


              ## print('kd_reprojkd_offset yyyz ',kd_reprojkd_offset)
              #############################






              # ### ### reprojected pose AFTER offset
              # reproj_rvec_pred2, reproj_tvec_pred2 = unpack_homo(reproj_kd_pose2)
              # show_axis(resized, reproj_rvec_pred2, reproj_tvec_pred2, cam_mat, dist, 20)
              

  # class CAM_POSE:
  #   def init(self):
  #     self.is_mm=True

  # class ARM_POSE:
  #   def init(self):
  #     self.is_mm=False
    
  #   def convert(self, cam_pose):
  #     if cam_pose.is_mm

  #   cam_pose = CAM_POSE()
              




          




          





          
          # if kd_tvec_pred is not None and ch_tvec_pred is not None:
          #   im = cv.imwrite('/home/hs3/catkin_ws/src/dvrk_record_video/result.png',resized)

          ###Whether show image
          cv.imshow(name_package, resized)
          pressed_key = cv.waitKey(1)
          if pressed_key == ord('q'): # User pressed 'q' to quit
            rospy.signal_shutdown("User requested to quit")
    r.sleep() # Keep loop running at the camera's FPS, by accounting the time used by any operations during the loop


if __name__ == '__main__':
    main(sys.argv)
