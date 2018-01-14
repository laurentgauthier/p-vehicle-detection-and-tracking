import cv2
import numpy as np
from skimage.feature import hog

def convert_color(image, cspace='RGB'):
    # Apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        converted_image = np.copy(image)

    return converted_image

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

# Define a function to draw bounding boxes
def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    image_copy = np.copy(image)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(image_copy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return image_copy

# Define a function to draw valid bounding boxes
def draw_valid_boxes(image, bboxes, past_bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    image_copy = np.copy(image)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if past_bboxes.is_valid_bbox(bbox):
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(image_copy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return image_copy
