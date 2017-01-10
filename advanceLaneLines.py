#!/home/pedgrfx/anaconda3/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from Camera import Camera
from Lanes import Lanes
from moviepy.editor import VideoFileClip
import glob

test_images = True
test_video1 = False
test_video2 = False
test_video3 = False

dump_frames = False

# Define the pipeline
def process_image(img):
    # Dump frames for debug?
    if dump_frames:
        filename = "./video_frames/frame" + str(lanes.frame_number) + ".jpg"
        mpimg.imsave(filename, img)

    # Apply distortion correction to the image
    undist = camera.undistort_image(img)

    # Dump frames for debug?
    if dump_frames:
        filename = "./video_frames/frame" + str(lanes.frame_number) + "_aundist" + ".jpg"
        mpimg.imsave(filename, undist)

    # Get the binary image using thresholding etc
    binary_img = camera.threshold_binary(undist)

    # Dump frames for debug?
    if dump_frames:
        filename = "./video_frames/frame" + str(lanes.frame_number) + "_binary" + ".jpg"
        mpimg.imsave(filename, binary_img)

    # Apply perspective transform 
    persp_img = camera.perspective_transform(binary_img)

    # Dump frames for debug?
    if dump_frames:
        filename = "./video_frames/frame" + str(lanes.frame_number) + "_persp" + ".jpg"
        mpimg.imsave(filename, persp_img)

    # Locate the lane lines
    lanes.locate_lanes(persp_img)

    # Dump frames for debug?
    if dump_frames:
        filename = "./video_frames/frame" + str(lanes.frame_number-1) + "_persp_lanes" + ".jpg"
        mpimg.imsave(filename, lanes.lane_debug_img)
        filename = "./video_frames/frame" + str(lanes.frame_number-1) + "_persp_left" + ".jpg"
        mpimg.imsave(filename, lanes.left_lane_img)
        filename = "./video_frames/frame" + str(lanes.frame_number-1) + "_persp_right" + ".jpg"
        mpimg.imsave(filename, lanes.right_lane_img)

    # Fit polynomials and set lane x/y arrays
    lanes.fit_lanes()

    # Check curvature sanity

    # Draw lines back onto road
    combined_img = lanes.draw_lanes(undist, persp_img, camera.Minv)

    # Dump frames for debug?
    if dump_frames:
        filename = "./video_frames/frame" + str(lanes.frame_number-1) + "_proc" + ".jpg"
        mpimg.imsave(filename, combined_img)

    return combined_img

# First lets calibrate the camera
camera = Camera(num_x_points=9, num_y_points=6, debug_mode=False)
camera.calibrate_camera("/home/pedgrfx/SDCND/AdvancedLaneFinding/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg")

# Un-distort test calibration image as demo that calibration is correct
img = mpimg.imread("/home/pedgrfx/SDCND/AdvancedLaneFinding/CarND-Advanced-Lane-Lines/camera_cal/calibration3.jpg")
undist = camera.undistort_image(img)
plt.imshow(undist)
plt.show()


if test_images:
    print("Running on test images...")
    #####################################
    # Run our pipeline on the test images
    #####################################
    images = glob.glob("/home/pedgrfx/SDCND/AdvancedLaneFinding/CarND-Advanced-Lane-Lines/test_images/test*.jpg")
    #images = glob.glob("/home/pedgrfx/SDCND/AdvancedLaneFinding/CarND-Advanced-Lane-Lines/video_frames/frame729.jpg")
    images.sort()
    #images = images[1:]

    # Setup the plot grid for test images
    plt.figure(figsize = (len(images),2))
    gs1 = gridspec.GridSpec(len(images),2)
    gs1.update(wspace=0.025, hspace=0.05)
    i=0

    for fname in images:
        # Define our Lanes object
        lanes = Lanes(debug_mode=True)

        print("Processing image {}".format(fname))

        # Next, let's read in a test image
        img = mpimg.imread(fname)
    
        # Process the image using our pipeline
        combined_img = process_image(img)
    
        # Plot the original image and the processed images
        ax1 = plt.subplot(gs1[i])
        plt.axis('off')
        ax1.imshow(img)
        ax2 = plt.subplot(gs1[i+1])
        plt.axis('off')
        ax2.imshow(combined_img)
        i += 2
    plt.show()
    
if test_video1:
    print("Running on test video1...")
    # Define our Lanes object
    lanes = Lanes(debug_mode=True)
    #####################################
    # Run our pipeline on the test video 
    #####################################
    clip = VideoFileClip("./project_video.mp4")
    output_video = "./project_video_processed.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

if test_video2:
    print("Running on test video2...")
    # Define our Lanes object
    lanes = Lanes(debug_mode=False)
    #####################################
    # Run our pipeline on the test video 
    #####################################
    clip = VideoFileClip("./challenge_video.mp4")
    output_video = "./challenge_video_processed.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

if test_video3:
    print("Running on test video3...")
    # Define our Lanes object
    lanes = Lanes(debug_mode=False)
    #####################################
    # Run our pipeline on the test video 
    #####################################
    clip = VideoFileClip("./harder_challenge_video.mp4")
    output_video = "./harder_challenge_video_processed.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)



