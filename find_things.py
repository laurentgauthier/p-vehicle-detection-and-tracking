import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

import utilities

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_things(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                hog_channel, cspace='RGB', spatial_size=(32, 32), hist_bins=32):

    bboxes = []

    # Make sure to scale image pixel values to [0.0, 1.0] range
    # WARNING: might be trouble depending on how the image was loaded as
    # some libraries/formats already do this.
    image = image.astype(np.float32)/255

    image_to_search = image[ystart:ystop,:,:]
    color_transformed_image_to_search = utilities.convert_color(image_to_search, cspace=cspace)
    if scale != 1:
        imshape = color_transformed_image_to_search.shape
        color_transformed_image_to_search = cv2.resize(color_transformed_image_to_search,
                                                       (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Separate all three channels in the image
    ch1 = color_transformed_image_to_search[:,:,0]
    ch2 = color_transformed_image_to_search[:,:,1]
    ch3 = color_transformed_image_to_search[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    if hog_channel == 0 or hog_channel == "ALL":
        hog1 = utilities.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 1 or hog_channel == "ALL":
        hog2 = utilities.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 2 or hog_channel == "ALL":
        hog3 = utilities.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract the HOG features for this patch
            if hog_channel == 0 or hog_channel == "ALL":
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == 1 or hog_channel == "ALL":
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == 2 or hog_channel == "ALL":
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

            if hog_channel == 0:
                hog_features = np.hstack((hog_feat1,))
            elif hog_channel == 1:
                hog_features = np.hstack((hog_feat2,))
            elif hog_channel == 2:
                hog_features = np.hstack((hog_feat3,))
            else:
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            sub_image = cv2.resize(color_transformed_image_to_search[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features for the patch
            spatial_features = utilities.bin_spatial(sub_image, size=spatial_size)
            hist_features = utilities.color_hist(sub_image, nbins=hist_bins)

            # Combine all the features
            combined_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            # Scale features and make a prediction
            test_features = X_scaler.transform(combined_features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return bboxes


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    import sys

    dist_pickle = pickle.load( open("svc_vehicles.p", "rb" ) )

    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    colorspace = dist_pickle["colorspace"]
    hog_channel = dist_pickle["hog_channel"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    image = mpimg.imread(sys.argv[1])

    ystart = 400
    ystop = 500

    scale = 0.75
    bboxes = find_things(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                         hog_channel, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)
    ystop = 550
    scale = 1.0
    bboxes += find_things(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                         hog_channel, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)
    ystop = 600
    scale = 1.5
    bboxes += find_things(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                         hog_channel, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins)

    out_image = utilities.draw_boxes(image, bboxes)

    plt.imshow(out_image)
    plt.show()
