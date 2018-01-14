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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

input_video_file = sys.argv[1]
output_video_file = sys.argv[2]

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

print(colorspace)
search_params = [
    (400, 700, 1.5),
    (400, 600, 1),
    (400, 500, 0.75),
    (400, 500, 0.65),
    (400, 450, 0.5),
    (400, 440, 0.4),
    (400, 700, 2.0),
]

class PastDetections:
    def __init__(self, max_depth):
        self.depth = max_depth
        self.past_bboxes = []

    def add_bboxes(self, latest_bboxes):
        self.past_bboxes.append(latest_bboxes)
        if len(self.past_bboxes) > self.depth:
            self.past_bboxes.pop(0)

    def draw_past_bboxes(self, image, color=(0, 0, 255), thick=6):
        out_image = np.copy(image)
        # Iterate through the old detected bboxes
        for bboxes in self.past_bboxes:
            for bbox in bboxes:
                cv2.rectangle(out_image, bbox[0], bbox[1], color, thick)
        return out_image

    def add_heat(self, heatmap):
        # Iterate through the old detected bboxes
        for bboxes in self.past_bboxes:
            for box in bboxes:
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    def is_valid_bbox(self, new_bbox):
        distance_threshold = 10 # pixels
        # Check that this bounding box is near another bounding box
        for bboxes in self.past_bboxes:
            for past_bbox in bboxes:
                past_center = ((past_bbox[0][0]+past_bbox[1][0])/2,(past_bbox[0][1]+past_bbox[1][1])/2)
                new_center = ((new_bbox[0][0]+new_bbox[1][0])/2,(new_bbox[0][1]+new_bbox[1][1])/2)
                distance = math.sqrt((past_center[0]-new_center[0])**2 + (past_center[1]-new_center[1])**2) 
                if distance < distance_threshold:
                    return True
        return False

def process_frame(image):
    global svc, X_scaler, orient, pix_per_cell, cell_per_block, colorspace, spatial_size, hist_bins, hog_channel
    global past_bboxes

    bboxes = []
    for index in range(4):
        ystart = search_params[index][0]
        ystop = search_params[index][1]
        scale = search_params[index][2]

        more_bboxes = find_things.find_things(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                              hog_channel, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)
        bboxes = bboxes + more_bboxes

    hot_bboxes = heat.find_hot_bboxes(bboxes, past_bboxes, heatmap_threshold=1)
    #out_image = utilities.draw_boxes(image, hot_bboxes)
    out_image = utilities.draw_valid_boxes(image, hot_bboxes, past_bboxes)
    #out_image = past_bboxes.draw_past_bboxes(image)

    # Keep a history of the recent past detections
    past_bboxes.add_bboxes(hot_bboxes)

    return out_image

past_bboxes = PastDetections(2)
video.process_clip(input_video_file, output_video_file, process_frame)
