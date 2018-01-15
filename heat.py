import numpy as np
from scipy.ndimage.measurements import label

# Add heat to heatmap
def add_heat(heatmap, bboxes):
    # Iterate through list of bboxes
    for box in bboxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

# Apply threshold to heat map
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

# Find bounding boxes based on labels
def find_labeled_bboxes(labeled_image, n_labels):
    bboxes = []
    # Iterate over the label values: [1..n_labels]
    for label_value in range(1, n_labels+1):
        # Find pixels by label value
        labeled_area = (labeled_image == label_value).nonzero()

        # Find the bounding box for this label
        labeled_area_x = np.array(labeled_area[1])
        labeled_area_y = np.array(labeled_area[0])
        bbox = ((np.min(labeled_area_x), np.min(labeled_area_y)),
                (np.max(labeled_area_x), np.max(labeled_area_y)))

        bboxes.append(bbox)

    return bboxes

def find_bboxes(bboxes, heatmap_threshold=2):
    # First draw a heatmap
    heatmap = np.zeros((720, 1280)) # FIXME Hardcoded image size
    add_heat(heatmap, bboxes)
    apply_threshold(heatmap, heatmap_threshold)

    # Then label the likely vehicles, and their bounding boxes
    labeled_image, n_labels = label(heatmap)
    labeled_bboxes = find_labeled_bboxes(labeled_image, n_labels)

    return labeled_bboxes

def find_hot_bboxes(bboxes, past_bboxes, heatmap_threshold=2):
    # First draw a heatmap
    heatmap = np.zeros((720, 1280)) # FIXME Hardcoded image size
    add_heat(heatmap, bboxes)
    #past_bboxes.add_heat(heatmap)
    apply_threshold(heatmap, heatmap_threshold)

    # Then label the likely vehicles, and draw bounding boxes for each
    labeled_image, n_labels = label(heatmap)
    labeled_bboxes = find_labeled_bboxes(labeled_image, n_labels)

    return labeled_bboxes
