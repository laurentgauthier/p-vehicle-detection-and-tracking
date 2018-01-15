import video
import glob
import sys
import cv2
import numpy as np
import pickle
import find_things
import utilities
import heat
import math
import past
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

input_video_file = sys.argv[1]
output_video_file = sys.argv[2]

# Retrieve the parameters required to use the LinearSVM classifier for
# the pickle dump created by training.py
dist_pickle = pickle.load(open("svc_vehicles.p", "rb" ))

svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
colorspace = dist_pickle["colorspace"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
hog_channel = dist_pickle["hog_channel"]

# Multi-scale search
search_params = [
    (400, 600, 1.5),
    (400, 550, 1),
    (400, 500, 0.75),
]

# Pipeline for one frame
def process_frame(image):
    global svc, X_scaler, orient, pix_per_cell, cell_per_block, colorspace, spatial_size, hist_bins, hog_channel

    bboxes = []
    # Search using three different regions (defined by ystart and ystop) and scale.
    for index in range(3):
        ystart = search_params[index][0]
        ystop = search_params[index][1]
        scale = search_params[index][2]

        bboxes += find_things.find_things(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                              hog_channel, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)

    hot_bboxes = heat.find_bboxes(bboxes, heatmap_threshold=1)
    out_image = utilities.draw_boxes(image, hot_bboxes)
    #out_image = utilities.draw_valid_boxes(image, hot_bboxes, past_bboxes)
    #out_image = past_bboxes.draw_past_bboxes(image)

    # Keep a history of the recent past detections
    #past_bboxes.add_bboxes(hot_bboxes)

    return out_image

#past_bboxes = past.PastDetections(3)
video.process_clip(input_video_file, output_video_file, process_frame)
