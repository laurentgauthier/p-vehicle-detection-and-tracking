import sys
import cv2
import numpy as np

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

