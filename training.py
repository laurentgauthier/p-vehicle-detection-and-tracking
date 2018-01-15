import os
import re
import time
import cv2
import numpy as np
import matplotlib.image as mpimg

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

import utilities

# Find PNG images in a directory (recursively)
def findImages(directory):
    rootDir = directory
    re_png = re.compile("^.*\.png$")
    images = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if re_png.match(fname):
                images.append('%s/%s' % (dirName, fname))
    return images

# Extract features from a list of images
def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, spatial_size=(32,32), hist_bins=32):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        feature_image = utilities.convert_color(image, cspace=cspace)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(utilities.get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = utilities.get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Get color features
        spatial_features = utilities.bin_spatial(image, size=spatial_size)
        hist_features = utilities.color_hist(image, nbins=hist_bins)

        # Combine all the features
        all_features = np.hstack((spatial_features, hist_features, hog_features))

        # Append the new feature vector to the features list
        features.append(all_features)

    # Return list of feature vectors
    return features


non_vehicles = findImages('data/non-vehicles')
vehicles = findImages('data/vehicles')

print('Size of the non-vehicles data set: %d' % len(non_vehicles))
print('Size of the vehicles data set: %d' % len(vehicles))

colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32

# Extract the features
vehicle_features = extract_features(vehicles, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)
non_vehicle_features = extract_features(non_vehicles, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins)

# Create an array stack of feature vectors
X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
# Scale the X feature vector
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations', pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature training vector length:', len(X_train))
print('    Feature test vector length:', len(X_test))

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

import pickle

model_dict = {
    'colorspace': colorspace,
    'orient': orient,
    'pix_per_cell': pix_per_cell,
    'cell_per_block': cell_per_block,
    'hog_channel': hog_channel,
    'spatial_size': spatial_size,
    'hist_bins': hist_bins,
    'scaler': X_scaler,
    'svc': svc
}

print(X_test[0].shape)
pickle_out = open("svc_vehicles.p","wb")
pickle.dump(model_dict, pickle_out)
pickle_out.close()
